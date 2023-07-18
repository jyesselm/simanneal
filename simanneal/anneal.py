from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
import copy
import math
import random
import time
from dataclasses import dataclass

from simanneal import logger


log = logger.get_logger("anneal")


@dataclass(frozen=True, order=True)
class AnnealerParams:
    steps: int
    tmax: float
    tmin: float
    copy_strategy: str = "deepcopy"
    pos: int = 0
    updates: int = 100
    output_file: str = ""


def round_figures(x, n):
    """Returns x rounded to n significant figures."""
    return round(x, int(n - math.ceil(math.log10(abs(x)))))


def time_string(seconds):
    """Returns time in seconds as a string formatted HHHH:MM:SS."""
    s = int(round(seconds))  # round to nearest second
    h, s = divmod(s, 3600)  # get hours and remainder
    m, s = divmod(s, 60)  # split remainder into minutes and seconds
    return "%4i:%02i:%02i" % (h, m, s)


class Annealer(object):
    """Performs simulated annealing by calling functions to calculate
    energy and make moves on a state.  The temperature schedule for
    annealing may be provided manually or estimated automatically.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, initial_state):
        self.params = AnnealerParams(50000, 25000, 2.5)
        self.best_state = None
        self.best_energy = None
        self.start = None
        self.state = self.copy_state(initial_state)

    @abc.abstractmethod
    def move(self):
        """Create a state change"""
        pass

    @abc.abstractmethod
    def energy(self):
        """Calculate state's energy"""
        pass

    def copy_state(self, state):
        """Returns an exact copy of the provided state
        Implemented according to self.copy_strategy, one of

        * deepcopy : use copy.deepcopy (slow but reliable)
        * slice: use list slices (faster but only works if state is list-like)
        * method: use the state's copy() method
        """
        if self.params.copy_strategy == "deepcopy":
            return copy.deepcopy(state)
        elif self.params.copy_strategy == "slice":
            return state[:]
        elif self.params.copy_strategy == "method":
            return state.copy()
        else:
            raise RuntimeError(
                "No implementation found for "
                + 'the self.copy_strategy "%s"' % self.params.copy_strategy
            )

    def update(self, *args, **kwargs):
        """Wrapper for internal update.

        If you override the self.update method,
        you can chose to call the self.default_update method
        from your own Annealer.
        """
        self.default_update(*args, **kwargs)

    def default_update(self, step, T, E, acceptance, improvement):
        """Default update, outputs to stderr.

        Prints the current temperature, energy, acceptance rate,
        improvement rate, elapsed time, and remaining time.

        The acceptance rate indicates the percentage of moves since the last
        update that were accepted by the Metropolis algorithm.  It includes
        moves that decreased the energy, moves that left the energy
        unchanged, and moves that increased the energy yet were reached by
        thermal excitation.

        The improvement rate indicates the percentage of moves since the
        last update that strictly decreased the energy.  At high
        temperatures it will include both moves that improved the overall
        state and moves that simply undid previously accepted moves that
        increased the energy by thermal excititation.  At low temperatures
        it will tend toward zero as the moves that can decrease the energy
        are exhausted and moves that would increase the energy are no longer
        thermally accessible."""

        elapsed = time.time() - self.start
        if step == 0:
            pass
        else:
            remain = (self.params.steps - step) * (elapsed / step)
            log.info(
                "\r%12.5f  %12.2f  %7.2f%%  %7.2f%%  %s  %s\r"
                % (
                    T,
                    E,
                    100.0 * acceptance,
                    100.0 * improvement,
                    time_string(elapsed),
                    time_string(remain),
                )
            )

    def anneal(self, params=None):
        """Minimizes the energy of a system by simulated annealing.

        Parameters
        state : an initial arrangement of the system

        Returns
        (state, energy): the best state and energy found.
        """
        if params is not None:
            self.params = params
        if params.output_file != "":
            f = open(params.output_file, "w")
            f.write("step,temp,energy,new_energy,best_energy,accepted\n")
        else:
            f = None

        step = 0
        self.start = time.time()

        # Precompute factor for exponential cooling from Tmax to Tmin
        if self.params.tmin <= 0.0:
            raise Exception(
                'Exponential cooling requires a minimum "\
                "temperature greater than zero.'
            )
        Tfactor = -math.log(self.params.tmax / self.params.tmin)

        # Note initial state
        T = self.params.tmax
        E = self.energy()
        log.debug(f"start energy: {round(E, 2)}")
        prevState = self.copy_state(self.state)
        prevEnergy = E
        self.best_state = self.copy_state(self.state)
        self.best_energy = E
        trials, accepts, improves = 0, 0, 0
        if self.params.updates > 0:
            updateWavelength = self.params.steps / self.params.updates
            self.update(step, T, E, None, None)
        log.debug(f"{'steps':5} {'cur':6} {'new':6}")
        # Attempt moves to new states
        while step < self.params.steps:
            step += 1
            T = self.params.tmax * math.exp(Tfactor * step / self.params.steps)
            dE = self.move()
            E += dE
            trials += 1
            if f is not None:
                f.write(f"{step},{round(T, 4)},{round(prevEnergy, 3)},{round(E, 3)},")
                f.write(f"{round(self.best_energy, 2)},")
            if dE > 0.0 and math.exp(-dE / T) < random.random():
                log.debug(f"{step:5} {round(prevEnergy, 2):6} {round(E, 2):6} REJ")
                if f is not None:
                    f.write("0\n")
                # Restore previous state
                self.state = self.copy_state(prevState)
                self.revert()
                E = prevEnergy
            else:
                log.debug(f"{step:5} {round(prevEnergy, 2):6} {round(E, 2):6} ACC")
                if f is not None:
                    f.write("1\n")
                # Accept new state and compare to best state
                accepts += 1
                if dE < 0.0:
                    improves += 1
                prevState = self.copy_state(self.state)
                prevEnergy = E
                if E < self.best_energy:
                    self.best_state = self.copy_state(self.state)
                    self.best_energy = E
            if self.params.updates > 1:
                if (step // updateWavelength) > ((step - 1) // updateWavelength):
                    self.update(step, T, E, accepts / trials, improves / trials)
                    trials, accepts, improves = 0, 0, 0

        self.state = self.copy_state(self.best_state)

        # Return best state and energy
        return self.best_state, self.best_energy

    def auto(self, minutes, steps=2000):
        """Explores the annealing landscape and
        estimates optimal temperature settings.

        Returns a dictionary suitable for the `set_schedule` method.
        """

        def run(T, steps):
            """Anneals a system at constant temperature and returns the state,
            energy, rate of acceptance, and rate of improvement."""
            E = self.energy()
            prevState = self.copy_state(self.state)
            prevEnergy = E
            accepts, improves = 0, 0
            for _ in range(steps):
                self.move()
                E = self.energy()
                dE = E - prevEnergy
                if dE > 0.0 and math.exp(-dE / T) < random.random():
                    self.state = self.copy_state(prevState)
                    self.revert()
                    E = prevEnergy
                else:
                    accepts += 1
                    if dE < 0.0:
                        improves += 1
                    prevState = self.copy_state(self.state)
                    prevEnergy = E
            return E, float(accepts) / steps, float(improves) / steps

        step = 0
        self.start = time.time()

        # Attempting automatic simulated anneal...
        # Find an initial guess for temperature
        T = 0.0
        E = self.energy()
        self.update(step, T, E, None, None)
        while T == 0.0:
            step += 1
            self.move()
            T = abs(self.energy() - E)

        # Search for Tmax - a temperature that gives 98% acceptance
        E, acceptance, improvement = run(T, steps)

        step += steps
        while acceptance > 0.98:
            T = round_figures(T / 1.5, 2)
            E, acceptance, improvement = run(T, steps)
            step += steps
            self.update(step, T, E, acceptance, improvement)
        while acceptance < 0.98:
            T = round_figures(T * 1.5, 2)
            E, acceptance, improvement = run(T, steps)
            step += steps
            self.update(step, T, E, acceptance, improvement)
        Tmax = T

        # Search for Tmin - a temperature that gives 0% improvement
        while improvement > 0.0:
            T = round_figures(T / 1.5, 2)
            E, acceptance, improvement = run(T, steps)
            step += steps
            self.update(step, T, E, acceptance, improvement)
        Tmin = T

        # Calculate anneal duration
        elapsed = time.time() - self.start
        duration = round_figures(int(60.0 * minutes * step / elapsed), 2)

        # Don't perform anneal, just return params
        return {"tmax": Tmax, "tmin": Tmin, "steps": duration, "updates": self.updates}
