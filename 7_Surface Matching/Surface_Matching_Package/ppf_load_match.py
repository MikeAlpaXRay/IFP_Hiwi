import cv2 as cv
import datetime
import os
from parameters import \
    model_path, scene_path, result_path,\
    relativeSamplingStep, relativeDistanceStep, numAngles, \
    relativeSceneSampleStep, relativeSceneDistance



#################################################################
# get Model .PLY Files from /Model with prefix (ACTIVE_)
#################################################################
models = []
possible_model_files = os.listdir(model_path)
for possible_model in possible_model_files:
    if possible_model.endswith(".ply") and\
            (possible_model.startswith("ACTIVE_") or possible_model.startswith("AKTIVE_")):
        models.append(possible_model)


#################################################################
# get Scene .PLY Files from /Scene with prefix (ACTIVE_)
#################################################################
possible_scene_files = os.listdir(scene_path)
for possible_scene in possible_scene_files:
    if possible_scene.endswith(".ply") and\
            (possible_scene.startswith("ACTIVE_") or possible_scene.startswith("AKTIVE_")):
        scene = possible_scene

# Number of Results used
N = 50


print('Prime detector...')
if numAngles == 0:
    detector = cv.ppf_match_3d_PPF3DDetector(relativeSamplingStep, relativeDistanceStep)
else:
    detector = cv.ppf_match_3d_PPF3DDetector(relativeSamplingStep, relativeDistanceStep, numAngles)

for model in models:
    now = datetime.datetime.now()
    print("####################### Current date and time: " + now.strftime("%Y %B%d - %H:%M:%S") +
          " #######################")
    print('Loading model...')
    pc = cv.ppf_match_3d.loadPLYSimple(model_path + "/%s" % model, 1)
    print(str(model) + " loaded")


    print('Training...')
    tick1 = cv.getTickCount()
    detector.trainModel(pc)
    tick2 = cv.getTickCount()
    print("Training complete in " + str((tick2-tick1)/ cv.getTickFrequency()) + "sec")

    print('Loading scene...')
    pcTest = cv.ppf_match_3d.loadPLYSimple(scene_path + "/%s" % scene, 1)
    print(str(scene) + " loaded")

    print('Matching...')
    tick1 = cv.getTickCount()
    results = detector.match(pcTest, relativeSceneSampleStep, relativeSceneDistance)
    tick2 = cv.getTickCount()
    print("Matching complete in " + str((tick2 - tick1) / cv.getTickFrequency()) + "sec")

    print('Performing ICP...')
    icp = cv.ppf_match_3d_ICP(100)
    _, results = icp.registerModelToScene(pc, pcTest, results[:N])

    now = datetime.datetime.now()
    print("####################### Current date and time: " + now.strftime("%Y %B%d - %H:%M:%S") +
          " #######################")



    # sort by resudial
    results.sort(key=lambda x: x.residual)


    for i, result in enumerate(results):
        if i == 0:
            print("\n####################### Model used #######################")
            print("\nPose to Model Index %d: NumVotes = %d, Residual = %f\n%s\n" % (result.modelIndex, result.numVotes,
                                                                                    result.residual, result.pose))
            print("\n##########################################################")

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
            if not numAngles == 0:
                parameter_file.write("numAngles:\t\t\t" + str(numAngles) + "\n")
            parameter_file.write("relativeSceneSampleStep:\t" + str(relativeSceneSampleStep) + "\n")
            parameter_file.write("relativeSceneDistance:\t\t" + str(relativeSceneDistance) + "\n")
            parameter_file.write("\n-- Pose to Model Index %d: NumVotes = %d, Residual = %f\n%s\n" % (result.modelIndex,
                                                                                                      result.numVotes,
                                                                                                      result.residual,
                                                                                                      result.pose))
            parameter_file.close()
        else:
            print("\n-- Pose to Model Index %d: NumVotes = %d, Residual = %f\n%s\n" % (result.modelIndex,
                                                                                       result.numVotes, result.residual,
                                                                                       result.pose))
