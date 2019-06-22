# Detection and Classification of Traffic Anomalies on Urban Arterial
## MEng Final Year Project 2019  
## Keidi Kapllani
### Imperial College London
#### Abstract
This final year project focuses on anomaly detection on urban arterial roads using
microscopic traffic variables. It is mainly concerned with the detection and classification
of speed anomalies in vehicles as well as clustering of these oscillations. The
proposed algorithms and methods and applied and evaluated on NGSIM trajectory
data taken on the US-101 highway. First, with the aim of pre-processing the data
for noise removal, sEMA filtering and Discrete Wavelet Transform denoising are
applied. The speed anomalies are then detected using an algorithm which has as
its core the Continuous Wavelet Transform. Following that, a multitude of clustering
algorithms is presented and evaluated for our specific problem. Finally, a novel
statistical method is used to detect breakpoints in the oscillations.
#### Code Usage

The code implements denoising for both sEMA smoothing and DWT denoising, anomaly detection & classification and clustering with breakpoint detection. All the relevant functions used are implemented in **utils.py**. The main functions are:

- **anomaly_detect()** : This functions detects anomalies in speed signals in a frame-by-frame fashion. Cars are windowed based on the time of entry in the observation zone. Frame-length and lane are selectable.  
    - INPUT: 
      - *data_smooth*: smooth data matrix for 1 time-period
      - *frame_length*: length of each frame to analyse
      - *start_frame*: index of starting frame (0 is first frame)
      - *num_frames*: number of frames (NOTE: Number fo frames should be proportional to frame length)
      - *start_lane*: Starting lane to consider (0- Lane 1 ... 4- Lane 5)
      - *num_lanes* : Number of lanes to analyse (MAX 5)

   -  OUTPUT: peaks_groups: All peaks detected in the form [Car ID, Peak Time, Peak Position , Peak Type]

- **cluster_peaks()** :  
  - INPUT: 
    - *peaks_groups*: All peaks detected in the form [Car ID, Peak Time, Peak Position , Peak Type]
  - OUTPUTS:
    - *wave_cluster*: List of clusters with peaks inside 
