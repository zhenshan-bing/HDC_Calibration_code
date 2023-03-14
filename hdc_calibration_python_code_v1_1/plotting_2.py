import numpy as np
import pickle
import scipy.spatial.transform.rotation as R
from scipy import interpolate
import math
from helper import angleDist
import matplotlib.pyplot as plt
import tikzplotlib
from helper import loadDataFromFile
import matplotlib.patches as patches

"""
w_ecd_to_conj = loadDataFromFile('data/model/weights/w_ecd_to_conj.obj')
plt.plot(np.arange(len(w_ecd_to_conj)), w_ecd_to_conj, color='orange')
plt.show()
"""

if __name__ == "__main__":
    print("Plots from Nitschke 2021 - Available Figures:")
    print("6.1a: Angular velocity in box environment")
    print("6.2:  HD error plot for the simulation in the box environment with a distal landmark")#done
    print("6.3:  HD error plot for the simulation in the box environment with a proximal landmark")#done
    print("6.4b: Angular velocity in maze environment")
    print("6.5a: HD error plot for the simulation in the maze environment with a proximal landmark")#done
    print("6.7a: Agent's route and real world setup: run 1")#done
    print("6.7b: Angular velocity real world experiment: run 1") #done
    print("6.7c: Agent's route and real world setup: run 2")#done
    print("6.7d: Angular velocity real world experiment: run 2") #done
    print("6.7e: Agent's route and real world setup: run 3")#done
    print("6.7f: Angular velocity real world experiment: run 3") #done
    print("6.8:  HD error plot for the simulation in the real world experiment: run 1")#done
    print("6.9:  HD error plot for the simulation in the real world experiment: run 2")#done
    print("6.10: HD error plot for the simulation in the real world experiment: run 3")#done
    print("7.2:  HD error plot from Arleo & Gerstner")#done
    print("------------------ NOTES ------------------")
    print("- All plots in `Section 5.3. Connection Weights` are created during the weights calculation in hdcCalibrConnectivity.py and can not be plotted here.")
    print("- The Figures B.1 & B.2 are created from controller.py (and tikzplotlib) and cannot be plotted here.")
    print("- The data for the HD error plots (6.2, 6.3, 6.5a, 6.8, 6.9, 6.10) is recorded from running controller.py")
    print("-------------------------------------------")
    terminate = False
    while not terminate:
        print("Enter figure number (e.g. '6.1') to plot it or close with 'exit': ", end="")
        fig_name = input()
        if fig_name == "6.1a":
            (thetas_data, _, t_episode, dt_robot) = loadDataFromFile('data/sim/sim_data_box_60sec.p')
            angVelocities = np.array(thetas_data)*((180/np.pi) * (1.0 / dt_robot))
            time_scale = (np.arange(0, t_episode, 0.05))
            plt.axhline(y=0, xmin=0, xmax=1, linestyle=':', color='black')
            plt.plot(time_scale, angVelocities, color='orange')
            plt.xlabel("time (s)")
            plt.ylabel("angular velocity (deg/s)")
            plt.show()

        elif fig_name == "6.2":
            csv_in = np.genfromtxt('data/sim/hd_error/hd_error_01_box_distal.csv', delimiter=';', skip_header=1)[:,
                     [0, 1, 2, 3, 4]]

            plt.plot(csv_in[:, [0]], csv_in[:, [1]], color=(0.12156862745098, 0.466666666666667, 0.705882352941177),
                     label='no calibration')  # blue
            plt.plot(csv_in[:, [0]], csv_in[:, [2]], color='#c95c5c', label='calibration with FGL')  # red
            plt.plot(csv_in[:, [0]], csv_in[:, [3]], color='#4d4d4d', label='calibration without FGL')  # gray
            plt.plot(csv_in[:, [0]], csv_in[:, [4]], color='#73ff73', label='simple feedback calibration')  # green
            plt.annotate('2.5s', xy=(2.7,0.1),color='#968d00')
            plt.annotate('9.5s', xy=(9.7,0.1),color='#968d00')
            plt.annotate('19.8s', xy=(20,0.1),color='#968d00')
            plt.axvline(x=2.5, ymin=0, ymax=1, linestyle=':', color='#968d00')
            plt.axvline(x=9.5, ymin=0, ymax=1, linestyle=':', color='#968d00')
            plt.axvline(x=19.8, ymin=0, ymax=1, linestyle=':', color='#968d00')
            plt.axhline(y=0, xmin=0, xmax=1, linestyle=':', color='black')

            plt.xlabel("time (s)")
            plt.ylabel("HD error (deg)")
            plt.legend()
            plt.show()

        elif fig_name == "6.3":
            csv_in = np.genfromtxt('data/sim/hd_error/hd_error_02_box_proximal.csv', delimiter=';', skip_header=1)[:, [0, 1, 2, 3, 4]]  # timestamp, quaternions(x,y,z,w), (av would be column 19)

            plt.plot(csv_in[:, [0]], csv_in[:, [1]], color=(0.12156862745098, 0.466666666666667, 0.705882352941177),
                     label='no calibration')  # blue
            plt.plot(csv_in[:, [0]], csv_in[:, [2]], color='#c95c5c', label='calibration with FGL')  # red
            plt.plot(csv_in[:, [0]], csv_in[:, [3]], color='#4d4d4d', label='calibration without FGL')  # gray
            plt.plot(csv_in[:, [0]], csv_in[:, [4]], color='#73ff73', label='simple feedback calibration')  # green
            plt.annotate('2.5s', xy=(2.7,0.3),color='#968d00')
            plt.annotate('9.8s', xy=(7.3,0.3),color='#968d00')
            plt.annotate('20.2s', xy=(20.4,0.3),color='#968d00')
            plt.annotate('10.7s', xy=(11, 0.3), color='#968d00')
            plt.axvline(x=2.5, ymin=0, ymax=1, linestyle=':', color='#968d00')
            plt.axvline(x=9.8, ymin=0, ymax=1, linestyle=':', color='#968d00')
            plt.axvline(x=20.2, ymin=0, ymax=1, linestyle=':', color='#968d00')
            plt.axvline(x=10.7, ymin=0, ymax=1, linestyle=':', color='#968d00')
            plt.axhline(y=0, xmin=0, xmax=1, linestyle=':', color='black')

            plt.xlabel("time (s)")
            plt.ylabel("HD error (deg)")
            plt.legend()
            plt.show()

        elif fig_name == "6.4b":
            (thetas_data, _, t_episode, dt_robot) = loadDataFromFile('data/sim/sim_data_maze_375sec.p')
            angVelocities = np.array(thetas_data)*((180/np.pi) * (1.0 / dt_robot))
            time_scale = (np.arange(0, t_episode, 0.05))
            plt.axhline(y=0, xmin=0, xmax=1, linestyle=':', color='black')
            plt.plot(time_scale, angVelocities, color='orange')
            plt.xlabel("time (s)")
            plt.ylabel("angular velocity (deg/s)")
            plt.show()

        elif fig_name == "6.5a":
            csv_in = np.genfromtxt('data/sim/hd_error/hd_error_03_maze.csv', delimiter=';', skip_header=1)[:, [0, 1, 2, 3, 4]]  # timestamp, quaternions(x,y,z,w), (av would be column 19)

            plt.plot(csv_in[:, [0]], csv_in[:, [1]], color=(0.12156862745098, 0.466666666666667, 0.705882352941177),
                     label='no calibration')  # blue
            plt.plot(csv_in[:, [0]], csv_in[:, [2]], color='#c95c5c', label='calibration with FGL')  # red
            plt.plot(csv_in[:, [0]], csv_in[:, [3]], color='#4d4d4d', label='calibration without FGL')  # gray
            plt.plot(csv_in[:, [0]], csv_in[:, [4]], color='#73ff73', label='simple feedback calibration')  # green

            plt.annotate('2.75s', xy=(5,67),color='#968d00')
            plt.annotate('359.5s', xy=(366.5,67),color='#968d00')
            plt.axvline(x=2.75, ymin=0, ymax=1, linestyle=':', color='#968d00')
            plt.axvline(x=364.5, ymin=0, ymax=1, linestyle=':', color='#968d00')
            plt.axhline(y=0, xmin=0, xmax=1, linestyle=':', color='black')

            plt.xlabel("time (s)")
            plt.ylabel("HD error (deg)")
            plt.legend()
            plt.show()

        elif fig_name == "6.7a":
            rw_run = "circle"

            pos_data_in = np.genfromtxt('data/rw_data/' + rw_run + '2_pos.csv', delimiter=',', skip_header=1)[:,
                          [0, 1, 2]]  # timestamp, x-pos, y-pos

            pos_data_in[:, [0]] = pos_data_in[:, [0]] - pos_data_in[[0], [0]]
            pos_data_in[:, [0]] = pos_data_in[:, [0]] / 1E9
            ref_points_x = [0, 0, 2000, 2000]
            ref_points_y = [0, 2000, 0, 2000]
            t_episode = math.floor(max(pos_data_in[:, [0]]))

            f = interpolate.interp1d(pos_data_in[:, 0], pos_data_in[:, 1], kind='linear')
            time_scale = (np.arange(0, t_episode, 0.05))

            x_pos_itp = f(time_scale)

            f = interpolate.interp1d(pos_data_in[:, 0], pos_data_in[:, 2], kind='linear')
            y_pos_itp = f(time_scale)

            plt.gca().invert_yaxis()
            plt.plot(x_pos_itp, y_pos_itp)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.scatter(ref_points_x, ref_points_y, marker='o', color='yellow')
            plt.scatter(ref_points_x, ref_points_y, marker='o', color='yellow')
            plt.scatter(1913.6139905, 580.953128846, marker='o', color='#c95c5c')
            plt.scatter(1955.11576566, 628.232647709, marker='o', color='#968d00')
            plt.scatter(1850, 650, marker='o', color='cyan')  ## 1955.11576566,628.232647709
            rect = patches.Rectangle((1350, 150), 1000, 1000,
                                     fill=False,
                                     color="cyan",
                                     linewidth=2)
            plt.gca().add_patch(rect)
            plt.annotate('Start', xy=(1900, 550))
            plt.annotate('End', xy=(1975, 625))
            plt.annotate('Landmark', xy=(1750, 718))
            plt.show()

        elif fig_name == "6.7b":
            (thetas_data, _, t_episode, dt_robot) = loadDataFromFile('data/sim/sim_data_rw_circle.p')
            angVelocities = np.array(thetas_data)*((180/np.pi) * (1.0 / dt_robot))
            time_scale = (np.arange(0, t_episode, 0.05))
            plt.axhline(y=0, xmin=0, xmax=1, linestyle=':', color='black')
            plt.plot(time_scale, angVelocities, color='orange')
            plt.xlabel("time (s)")
            plt.ylabel("angular velocity (deg/s)")
            plt.show()

        elif fig_name == "6.7c":
            rw_run = "loops"

            pos_data_in = np.genfromtxt('data/rw_data/' + rw_run + '2_pos.csv', delimiter=',', skip_header=1)[:,
                          [0, 1, 2]]  # timestamp, x-pos, y-pos

            pos_data_in[:, [0]] = pos_data_in[:, [0]] - pos_data_in[[0], [0]]
            pos_data_in[:, [0]] = pos_data_in[:, [0]] / 1E9
            ref_points_x = [0, 0, 2000, 2000]
            ref_points_y = [0, 2000, 0, 2000]
            t_episode = math.floor(max(pos_data_in[:, [0]]))

            f = interpolate.interp1d(pos_data_in[:, 0], pos_data_in[:, 1], kind='linear')
            time_scale = (np.arange(0, t_episode, 0.05))

            x_pos_itp = f(time_scale)

            f = interpolate.interp1d(pos_data_in[:, 0], pos_data_in[:, 2], kind='linear')
            y_pos_itp = f(time_scale)

            plt.gca().invert_yaxis()
            plt.plot(x_pos_itp, y_pos_itp)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.scatter(ref_points_x, ref_points_y, marker='o', color='yellow')
            plt.scatter(ref_points_x, ref_points_y, marker='o', color='yellow')
            plt.scatter(1350, 400, marker='o', color='cyan')  ## 1955.11576566,628.232647709
            rect = patches.Rectangle((850, -100), 1000, 1000,
                                     fill=False,
                                     color="cyan",
                                     linewidth=2)
            plt.gca().add_patch(rect)
            plt.scatter(1849.0311177,369.165011422, marker='o', color='#968d00')
            plt.scatter(1993.64464302,493.354088737, marker='o', color='#c95c5c')#968d00

            plt.annotate('Start', xy=(1870, 368))
            plt.annotate('End', xy=(2010, 500))
            plt.annotate('Landmark', xy=(1300, 356))
            plt.show()

        elif fig_name == "6.7d":
            (thetas_data, _, t_episode, dt_robot) = loadDataFromFile('data/sim/sim_data_rw_loops.p')
            angVelocities = np.array(thetas_data) * ((180 / np.pi) * (1.0 / dt_robot))
            time_scale = (np.arange(0, t_episode, 0.05))
            plt.axhline(y=0, xmin=0, xmax=1, linestyle=':', color='black')
            plt.plot(time_scale, angVelocities, color='orange')
            plt.xlabel("time (s)")
            plt.ylabel("angular velocity (deg/s)")
            plt.show()

        elif fig_name == "6.7e":
            rw_run = "cross"

            pos_data_in = np.genfromtxt('data/rw_data/' + rw_run + '2_pos.csv', delimiter=',', skip_header=1)[:,
                          [0, 1, 2]]  # timestamp, x-pos, y-pos

            pos_data_in[:, [0]] = pos_data_in[:, [0]] - pos_data_in[[0], [0]]
            pos_data_in[:, [0]] = pos_data_in[:, [0]] / 1E9
            ref_points_x = [0, 0, 2000, 2000]
            ref_points_y = [0, 2000, 0, 2000]
            t_episode = math.floor(max(pos_data_in[:, [0]]))

            f = interpolate.interp1d(pos_data_in[:, 0], pos_data_in[:, 1], kind='linear')
            time_scale = (np.arange(0, t_episode, 0.05))

            x_pos_itp = f(time_scale)

            f = interpolate.interp1d(pos_data_in[:, 0], pos_data_in[:, 2], kind='linear')
            y_pos_itp = f(time_scale)

            plt.gca().invert_yaxis()
            plt.plot(x_pos_itp, y_pos_itp)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.scatter(ref_points_x, ref_points_y, marker='o', color='yellow')
            plt.scatter(ref_points_x, ref_points_y, marker='o', color='yellow')
            plt.scatter(1250, 500, marker='o', color='cyan')  ## 1955.11576566,628.232647709
            rect = patches.Rectangle((850, -100), 1000, 1000,
                                     fill=False,
                                     color="cyan",
                                     linewidth=2)
            plt.gca().add_patch(rect)
            plt.scatter(1825.58679604,443.92595205, marker='o', color='#968d00')
            plt.scatter(2053.06969569,561.384026705, marker='o', color='#c95c5c')#968d00

            plt.annotate('Start', xy=(1860, 435))
            plt.annotate('End', xy=(2068, 530))
            plt.annotate('Landmark', xy=(1280, 495))
            plt.show()

        elif fig_name == "6.7f":
            (thetas_data, _,t_episode, dt_robot) = loadDataFromFile('data/sim/sim_data_rw_cross.p')
            angVelocities = np.array(thetas_data) * ((180 / np.pi) * (1.0 / dt_robot))
            time_scale = (np.arange(0, t_episode, 0.05))
            plt.axhline(y=0, xmin=0, xmax=1, linestyle=':', color='black')
            plt.plot(time_scale, angVelocities, color='orange')
            plt.xlabel("time (s)")
            plt.ylabel("angular velocity (deg/s)")
            plt.show()

        elif fig_name == "6.8":
            csv_in = np.genfromtxt('data/sim/hd_error/hd_error_04_rw_1.csv', delimiter=';', skip_header=1)[:, [0, 1, 2, 3]]  # timestamp, quaternions(x,y,z,w), (av would be column 19)

            plt.plot(csv_in[:, [0]], csv_in[:, [1]], color=(0.12156862745098, 0.466666666666667, 0.705882352941177),
                     label='no calibration')  # blue
            plt.plot(csv_in[:, [0]], csv_in[:, [2]], color='#c95c5c', label='calibration with FGL')  # red
            plt.plot(csv_in[:, [0]], csv_in[:, [3]], color='#4d4d4d', label='calibration without FGL')  # gray

            plt.annotate('6.7s', xy=(6.9,0.7),color='#968d00')
            plt.annotate('7.8s', xy=(8,0.7),color='#968d00')
            plt.annotate('22.6s', xy=(22.8,0.7),color='#968d00')

            plt.axvline(x=6.7, ymin=0, ymax=1, linestyle=':', color='#968d00')
            plt.axvline(x=7.8, ymin=0, ymax=1, linestyle=':', color='#968d00')
            plt.axvline(x=22.6, ymin=0, ymax=1, linestyle=':', color='#968d00')

            plt.axhline(y=0, xmin=0, xmax=1, linestyle=':', color='black')

            plt.xlabel("time (s)")
            plt.ylabel("HD error (deg)")
            plt.legend()
            plt.show()

        elif fig_name == "6.9":
            csv_in = np.genfromtxt('data/sim/hd_error/hd_error_05_rw_2.csv', delimiter=';', skip_header=1)[:, [0, 1, 2, 3]]  # timestamp, quaternions(x,y,z,w), (av would be column 19)

            plt.plot(csv_in[:, [0]], csv_in[:, [1]], color=(0.12156862745098, 0.466666666666667, 0.705882352941177),
                     label='no calibration')  # blue
            plt.plot(csv_in[:, [0]], csv_in[:, [2]], color='#c95c5c', label='calibration with FGL')  # red
            plt.plot(csv_in[:, [0]], csv_in[:, [3]], color='#4d4d4d', label='calibration without FGL')  # gray

            plt.annotate('3.2s', xy=(3.5,8.5),color='#968d00')
            plt.annotate('17.3s', xy=(17.5,8.5),color='#968d00')
            plt.annotate('32.0s', xy=(32.2,8.5),color='#968d00')
            plt.annotate('40.35s', xy=(40.35,8.5),color='#968d00')
            plt.annotate('42.1s', xy=(42.3,8.5),color='#968d00')

            plt.axvline(x=3.2, ymin=0, ymax=1, linestyle=':', color='#968d00')
            plt.axvline(x=17.3, ymin=0, ymax=1, linestyle=':', color='#968d00')
            plt.axvline(x=32.0, ymin=0, ymax=1, linestyle=':', color='#968d00')
            plt.axvline(x=40.35, ymin=0, ymax=1, linestyle=':', color='#968d00')
            plt.axvline(x=42.1, ymin=0, ymax=1, linestyle=':', color='#968d00')
            plt.axhline(y=0, xmin=0, xmax=1, linestyle=':', color='black')

            plt.xlabel("time (s)")
            plt.ylabel("HD error (deg)")
            plt.legend()
            plt.show()

        elif fig_name == "6.10":
            csv_in = np.genfromtxt('data/sim/hd_error/hd_error_06_rw_3.csv', delimiter=';', skip_header=1)[:, [0, 1, 2, 3]]  # timestamp, quaternions(x,y,z,w), (av would be column 19)

            plt.plot(csv_in[:, [0]], csv_in[:, [1]], color=(0.12156862745098, 0.466666666666667, 0.705882352941177),
                     label='no calibration')  # blue
            plt.plot(csv_in[:, [0]], csv_in[:, [2]], color='#c95c5c', label='calibration with FGL')  # red
            plt.plot(csv_in[:, [0]], csv_in[:, [3]], color='#4d4d4d', label='calibration without FGL')  # gray

            plt.annotate('1.55s', xy=(1.75,12),color='#968d00')
            plt.annotate('13.5s', xy=(13.7,12),color='#968d00')
            plt.annotate('24.05s', xy=(24.25,12),color='#968d00')
            plt.annotate('35.0s', xy=(35.2,12),color='#968d00')
            plt.annotate('37.75s', xy=(37.95,12),color='#968d00')

            plt.axvline(x=1.55, ymin=0, ymax=1, linestyle=':', color='#968d00')
            plt.axvline(x=13.5, ymin=0, ymax=1, linestyle=':', color='#968d00')
            plt.axvline(x=24.05, ymin=0, ymax=1, linestyle=':', color='#968d00')
            plt.axvline(x=35.0, ymin=0, ymax=1, linestyle=':', color='#968d00')
            plt.axvline(x=37.75, ymin=0, ymax=1, linestyle=':', color='#968d00')

            plt.axhline(y=0, xmin=0, xmax=1, linestyle=':', color='black')

            plt.xlabel("time (s)")
            plt.ylabel("HD error (deg)")
            plt.legend()
            plt.show()

        elif fig_name == "7.2":
            csv_in = np.genfromtxt('data/sim/hd_error/hd_error_arleo_gerstner.csv', delimiter=';', skip_header=1)[:,
                     [0, 1, 2, 3]]  # timestamp, quaternions(x,y,z,w), (av would be column
            plt.plot(csv_in[:, [0]], csv_in[:, [1]], color=(0.12156862745098, 0.466666666666667, 0.705882352941177),
                     label='without calibration')  # blue
            plt.plot(csv_in[:, [2]], csv_in[:, [3]], color='#c95c5c', label='with calibration')  #
            plt.xlabel("time (s)")
            plt.ylabel("HD error (deg)")
            plt.legend()
            plt.show()

        elif fig_name == "exit":
            terminate = True
        else:
            print("Figure '{}' unknown, exit with 'exit'".format(fig_name))