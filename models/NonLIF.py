from models.BaseNeuronModel import BaseNeuronModel


class NonLIF(BaseNeuronModel):
    def step(self, input_current: float):
        v = self.voltages[-1]

        # check if voltage is above threshold
        if v >= self.v_th:
            self.spike_times += [len(self.voltages) - 1]
            v = self.v_reset

        # calculate membrane voltage change
        dv = (self.dt / self.tau_m) * (input_current / self.R)

        v += dv

        self.voltages += [v]
