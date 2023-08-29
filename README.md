# ARM_analysis

This repo performs a precipitation-buoyancy analysis using [ARMBE](https://www.arm.gov/capabilities/science-data-products/vaps/armbe) data. 
The following steps are performed:
1. The atmosphere is divided into a boundary layer and a lower-free troposphere:
   1. The highest valid pressure level is treated as the surface (psfc).
   2. A fixed mid-troposphere level, pmid (default = 500 hPa) is used.
   3. The boundary layer is assigned a fixed proportion of psfc - pmid.
   4. The boundary layer and lower-free tropospheric levels are used to computed coefficients $w_B$ and $w_L = 1-w_B$
3. The equivalent potential temperature, $\theta_e$ is calculated and average over the boundary layer. This variable is termed $\theta_{eB}$
4. Similarly, the lower-tropospheric averaged equivalent potential temperature $\theta_{eL}$ and saturation equivalent potential temperature $\theta^*_{eL}$ are computed.
5. The instability, subsaturation and buoyancy components are computed following [Ahmed et al. 2020](https://journals.ametsoc.org/view/journals/atsc/77/6/JAS-D-19-0227.1.xml):
6. Each buoyancy environment is then matched to its typical precipitation value. 


