import cv2 as cv
import datetime
import os

#################################################################
# Paths
#################################################################
model_path = "01_Models"
scene_path = "02_Scene"
result_path = "03_Result"


#################################################################
# get Model .PLY Files from /Model with prefix (ACTIVE_)
#################################################################
models = []
possible_model_files = os.listdir(model_path)
for possible_model in possible_model_files:
    if possible_model.endswith(".ply") and possible_model.startswith("ACTIVE_"):
        models.append(possible_model)


#################################################################
# get Scene .PLY Files from /Scene with prefix (ACTIVE_)
#################################################################
possible_scene_files = os.listdir(scene_path)
for possible_scene in possible_scene_files:
    if possible_scene.endswith(".ply") and possible_scene.startswith("ACTIVE_"):
        scene = possible_scene


N = 2

relativeSamplingStep = 0.03
# Sampling distance relative to the object's diameter.
# Models are first sampled uniformly in order to improve efficiency.
# Decreasing this value leads to a denser model, and a more accurate pose estimation but the larger the model, the slower the training.
# Increasing the value leads to a less accurate pose computation but a smaller model and faster model generation and matching.
# Beware of the memory consumption when using small values.
relativeDistanceStep = 0.05
# The discretization distance of the point pair distance relative to the model's diameter.
# This value has a direct impact on the hashtable.
# Using small values would lead to too fine discretization, and thus ambiguity in the bins of hashtable.
# Too large values would lead to no discrimination over the feature vectors and different point pair features would be assigned to the same bin.
# This argument defaults to the value of RelativeSamplingStep.
# For noisy scenes, the value can be increased to improve the robustness of the matching against noisy points.
numAngles = 50
# Set the discretization of the point pair orientation as the number of subdivisions of the angle.
# This value is the equivalent of RelativeDistanceStep for the orientations.
# Increasing the value increases the precision of the matching but decreases the robustness against incorrect normal directions.
# Decreasing the value decreases the precision of the matching but increases the robustness against incorrect normal directions.
# For very noisy scenes where the normal directions can not be computed accurately, the value can be set to 25 or 20.

relativeSceneSampleStep = 1/2
# The ratio of scene points to be used for the matching after sampling with relativeSceneDistance.
# For example, if this value is set to 1.0/5.0, every 5th point from the scene is used for pose estimation.
# This parameter allows an easy trade-off between speed and accuracy of the matching.
# Increasing the value leads to less points being used and in turn to a faster but less accurate pose computation.
# Decreasing the value has the inverse effect.
relativeSceneDistance = 0.05
# Set the distance threshold relative to the diameter of the model.
# This parameter is equivalent to relativeSamplingStep in the training stage.
# This parameter acts like a prior sampling with the relativeSceneSampleStep parameter.


print('Prime detector...')
detector = cv.ppf_match_3d_PPF3DDetector(relativeSamplingStep, relativeDistanceStep)
#detector = cv.ppf_match_3d_PPF3DDetector(relativeSamplingStep, relativeDistanceStep, numAngles)

for model in models:
    print('Loading model...')
    pc = cv.ppf_match_3d.loadPLYSimple(model_path + "/%s" % model, 1)

    now = datetime.datetime.now()
    print("####################### Current date and time: " + now.strftime("%Y %B%d - %H:%M:%S") + " #######################")

    print('Training...')
    tick1 = cv.getTickCount()
    detector.trainModel(pc)
    tick2 = cv.getTickCount()
    print("Training complete in " + str((tick2-tick1)/ cv.getTickFrequency()) + "sec")

    print('Loading scene...')
    tick1 = cv.getTickCount()
    pcTest = cv.ppf_match_3d.loadPLYSimple(scene_path + "/%s" % scene, 1)
    tick2 = cv.getTickCount()
    print("Scene loading complete in " + str((tick2-tick1)/ cv.getTickFrequency()) + "sec")

    print('Matching...')
    tick1 = cv.getTickCount()
    results = detector.match(pcTest, relativeSceneSampleStep, relativeSceneDistance)
    tick2 = cv.getTickCount()
    print("Matching complete in " + str((tick2 - tick1) / cv.getTickFrequency()) + "sec")

    print('Performing ICP...')
    icp = cv.ppf_match_3d_ICP(100)
    _, results = icp.registerModelToScene(pc, pcTest, results[:N])

    now = datetime.datetime.now()
    print("####################### Current date and time: " + now.strftime("%Y %B%d - %H:%M:%S") + " #######################")



    # sort by resudial
    results.sort(key=lambda x: x.residual)


    for i, result in enumerate(results):
        print(result.residual)
    for i, result in enumerate(results):
        print("\n-- Pose to Model Index %d: NumVotes = %d, Residual = %f\n%s\n" % (result.modelIndex, result.numVotes, result.residual, result.pose))



        if i == 0:
            print("\n-- Pose to Model Index %d: NumVotes = %d, Residual = %f\n%s\n" % (result.modelIndex, result.numVotes, result.residual, result.pose))

            pct = cv.ppf_match_3d.transformPCPose(pc, result.pose)

            now = datetime.datetime.now()
            timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
            result_file_name = result_path + "/" + timestamp + "___%s---%s" % (model[7:-4], scene[7:-4])
            cv.ppf_match_3d.writePLY(pct, result_file_name + ".ply")




            parameter_file = open(result_file_name + ".txt", "w+")
            parameter_file.write("Parameters:\n")
            parameter_file.write("Model:\t" + model + "\n")
            parameter_file.write("Scene:\t" + scene + "\n")
            parameter_file.write("relativeSamplingStep:\t\t" + str(relativeSamplingStep) + "\n")
            parameter_file.write("relativeDistanceStep:\t\t" + str(relativeDistanceStep) + "\n")
            #parameter_file.write("numAngles:\t\t\t" + str(numAngles) + "\n")
            parameter_file.write("relativeSceneSampleStep:\t" + str(relativeSceneSampleStep) + "\n")
            parameter_file.write("relativeSceneDistance:\t\t" + str(relativeSceneDistance) + "\n")
            parameter_file.close()
