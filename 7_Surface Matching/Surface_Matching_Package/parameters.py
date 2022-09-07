#################################################################
# Paths
#################################################################
model_path = "01_Models"
scene_path = "02_Scene"
result_path = "03_Result"
scene_compare_path = "04_Scene_Compare_Pose"


#################################################################
# Sampling distance relative to the object's diameter.
# Models are first sampled uniformly in order to improve efficiency.
# Decreasing this value leads to a denser model, and a more accurate pose estimation but the larger the model, the slower the training.
# Increasing the value leads to a less accurate pose computation but a smaller model and faster model generation and matching.
# Beware of the memory consumption when using small values.
#################################################################
relativeSamplingStep = 0.1


#################################################################
# The discretization distance of the point pair distance relative to the model's diameter.
# This value has a direct impact on the hashtable.
# Using small values would lead to too fine discretization, and thus ambiguity in the bins of hashtable.
# Too large values would lead to no discrimination over the feature vectors and different point pair features would be assigned to the same bin.
# This argument defaults to the value of RelativeSamplingStep.
# For noisy scenes, the value can be increased to improve the robustness of the matching against noisy points.
#################################################################
relativeDistanceStep = 0.5


#################################################################
# If 0 default is used
# Set the discretization of the point pair orientation as the number of subdivisions of the angle.
# This value is the equivalent of RelativeDistanceStep for the orientations.
# Increasing the value increases the precision of the matching but decreases the robustness against incorrect normal directions.
# Decreasing the value decreases the precision of the matching but increases the robustness against incorrect normal directions.
# For very noisy scenes where the normal directions can not be computed accurately, the value can be set to 25 or 20.
#################################################################
numAngles = 10


#################################################################
# The ratio of scene points to be used for the matching after sampling with relativeSceneDistance.
# For example, if this value is set to 1.0/5.0, every 5th point from the scene is used for pose estimation.
# This parameter allows an easy trade-off between speed and accuracy of the matching.
# Increasing the value leads to less points being used and in turn to a faster but less accurate pose computation.
# Decreasing the value has the inverse effect.
#################################################################
relativeSceneSampleStep = 1/2


#################################################################
# Set the distance threshold relative to the diameter of the model.
# This parameter is equivalent to relativeSamplingStep in the training stage.
# This parameter acts like a prior sampling with the relativeSceneSampleStep parameter.
#################################################################
relativeSceneDistance = 0.03