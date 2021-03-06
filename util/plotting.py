import matplotlib.pyplot as plt
import numpy as np


def plot_voltage_trace(voltage_trace, v_th, dt, sim_length):
    sim_range = np.arange(0, sim_length * dt, dt)

    # plot voltage trace
    plt.plot(sim_range, voltage_trace, "b")

    plt.axhline(v_th, 0, 1, color="k", ls="--")
    plt.xlabel("Time (ms)")
    plt.ylabel("V (mV)")
    plt.legend(
        ["Membrane\npotential", r"Threshold V$_{\mathrm{th}}$"], loc=[1.05, 0.75]
    )
    plt.ylim([-80, -40])


def plot_voltage_trace_with_sra(voltage_trace, v_th, dt, sim_length, sra):
    sim_range = np.arange(0, sim_length * dt, dt)

    fig, axs = plt.subplots(nrows=2, ncols=1)

    # plot voltage trace
    axs[0].plot(sim_range, voltage_trace, "b")
    axs[0].axhline(v_th, 0, 1, color="k", ls="--")
    axs[0].set_ylim([-80, -40])
    axs[0].set_ylabel("V (mV)")
    axs[0].legend(
        ["Membrane\npotential", r"Threshold V$_{\mathrm{th}}$"], loc=[1.05, 0.3]
    )

    # plot spike rate adaption values
    axs[1].plot(sim_range, sra, "r")
    axs[1].set_ylabel("SRA (mV)")
    axs[1].legend(["Spike rate\nadaption"], loc=[1.05, 0.5])

    plt.xlabel("Time (ms)")


def plot_spikes(spike_times, dt, sim_length):

    plt.eventplot(np.array(spike_times) * dt, linelengths=4, colors="k")
    plt.xlim([0, sim_length * dt])

    plt.yticks([])

    plt.title("Spike Times")
    plt.xlabel("Time (ms)")
    plt.ylabel("Spike")
