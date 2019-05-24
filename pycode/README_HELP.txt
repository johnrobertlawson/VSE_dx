(Pdb) obj_obs.iloc[0]
area                                                                           441
bbox_area                                                                      750
case_code                                                                 20160331
centroid_col                                                               41.3265
centroid_lat                                                                32.341
centroid_lon                                                              -90.5759
centroid_row                                                               37.2426
conv_mode                                                                 cellular
convex_area                                                                    566
domain                                                                      nexrad
dx                                                                               1
eccentricity                                                              0.665265
equivalent_diameter                                                         23.696
extent                                                                       0.588
fpath_save                       /Volumes/LaCie/VSE_dx/object_instances/2016033...
index                                                                           26
init_code                                                                      NaN
label                                                                            5
lead_time                                                                        0
leadtime_group                                                          first_hour
level_0                                                                          0
longaxis_km                                                                29.0902
lowrot_angle_from_centroid                                                 243.177
lowrot_distance_from_centroid                                               10.357
lowrot_exceed_ID_0                                                            True
lowrot_exceed_ID_0_val                                                      0.0017
lowrot_exceed_ID_1                                                            True
lowrot_exceed_ID_1_val                                                      0.0041
lowrot_exceed_ID_2                                                           False
lowrot_exceed_ID_2_val                                                      0.0078
                                                       ...
midrot_exceed_ID_0                                                            True
midrot_exceed_ID_0_val                                                      0.0019
midrot_exceed_ID_1                                                            True
midrot_exceed_ID_1_val                                                      0.0044
midrot_exceed_ID_2                                                            True
midrot_exceed_ID_2_val                                                      0.0081
midrot_exceed_ID_3                                                           False
midrot_exceed_ID_3_val                                                      0.0135
min_col                                                                         25
min_intensity                                                              45.0828
min_lowrot                                                             -0.00213865
min_midrot                                                             -0.00362584
min_row                                                                         26
min_updraught                                                                  NaN
nlats                                                                          321
nlons                                                                          321
perimeter                                                                  132.882
prod_code                                                           nexrad_1km_obs
qlcsness                                                                  -1.34377
ratio                                                                     0.746607
resolution                                                                   EE1km
test_gridsize                                                                  NaN
time                                                           2016-03-31 21:20:00
ud_angle_from_centroid                                                         NaN
ud_distance_from_centroid                                                      NaN
weighted_centroid_col                                                      41.4328
weighted_centroid_lat                                                      32.3401
weighted_centroid_lon                                                     -90.5747
weighted_centroid_row                                                      37.1319
megaframe_idx                                                                  426
Name: 426, Length: 81, dtype: object





# Counts of objects:


fcst_fmts = ("d02_1km","d01_3km")
obs_fmts = ("nexrad_3km","nexrad_1km")
all_fmts = list(fcst_fmts) + list(obs_fmts)

(Pdb) df_list[0].shape
(113210, 80)
(Pdb) df_list[1].shape
(94984, 80)
(Pdb) df_list[2].shape
(11201, 72)
(Pdb) df_list[3].shape
(11284, 72)
