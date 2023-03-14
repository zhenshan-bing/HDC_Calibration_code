Installation:
 * Required software:
   - Python 3 (Python 2 may work as well)
   - NumPy (pip install numpy)
   - SciPy (pip install scipy)
   - PyBullet (pip install pybullet)
   - tqdm (pip install tqdm)
 * Recommended software:
   - PyCuda for GPU acceleration (https://wiki.tiker.net/PyCuda/Installation/) (if a GPU with CUDA support is available)
 * Set the backend (NumPy or CUDA) in network.py

Scripts:
 - controller.py: run the simulation. Environment/dataset and simulation parameters can be set on the top of that file. Additionally, it can be chosen which calibration model is used. (used to create error & angular veclocity plots in Nitschke 2021)
 - generate_sim_data_file.py: run pybullet and save all the changes in angle & agent states (positions, orientations) for every timestep in a file readable by controller.py
  - generate_sim_data_from_rw.py: read out csv-files that contain the data from the real world experiment and convert all the changes in angle & agent states (positions, orientations) for every timestep into a file readable by controller.py
 - hdcCalibrConnectivity.py: Calculates all weights between the components of the calibration circuit (HD,ECD,CONJ,ACD,CONJ2), plots the results & saves them as files in path "data\model\weights"
 - plotting.py: generates most of the figures in the bachelor thesis of El-Sewisy 2020
 - plotting_2.py: generates the plots in Section 6 "Experiments" in the master's thesis of Nitschke 2021

Functionality:
 - hdc_template.py: generate the whole HDC network (basis HD model and calibration circuit components) according to parameters defined in params.py
 - hdcAttractorConnectivity: generate connections inside the HDC attractor network given the parameter lambda
 - hdcNetwork.py: provides the functions to build the whole HDC network (to add and connect layers to a topology), to initialize/simulate/stimulate layers of the HDC network
 - hdcOptimizeAttractor.py: contains the procedure used to find the value for lambda yielding the best weight function for the connections within the HDC attractor
 - hdcOptimizeShiftLayers.py: contains the procedure used to generate the plots for shifting (Figures 2.11, 2.12) as well as finding the factor for the angular velocity -> stimulus function
 - hdcTargetPeak.py: Contains the target peak function for HD cells, ECD cells and ACD cells, as well as the function describing the cross-section of the 2D activity peak for the conjunctive cells
 - helper.py: helper functions (decode attractor network, distance between angles, ...)
 - import_ros.py: Only used by El-Sewisy 2020 (import ROS messages from real world experiments and stores it into a pickle file to read it using controller.py)
 - network.py: proxy for selecting one of the network backends:
   - network_cuda.py
   - network_numpy.py
 - neuron.py: neuron model parameters
 - params.py: parameters for the network (number of neurons, lambda)
 - placeEncoding.py: Contains the position encoding calculations & the storing/restoring of allocentric cue directions depending on the chosen mode (first glance learning "on" or "off")
 - polarPlotter.py: live visualization of the whole HDC network (also used to plot the whole calibration circuit in Nitschke 2021)
 - pybullet_environment.py: pybullet environment class, braitenberg controller
 - stresstest.py: performance testing
  