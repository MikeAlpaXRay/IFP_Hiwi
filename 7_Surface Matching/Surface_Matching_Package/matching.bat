:: Activates Python Enviroment
call conda activate OpenCVSurfaceMatching
:: Change to personal Filepath
::Parameter %* relativeSamplingStep, relativeDistanceStep, numAngles, relativeSceneSampleStep, relativeSceneDistance
python "E:\Cloud\OneDrive - bwedu\Studium\Master\HiWi\IFP\7_Surface Matching\Surface_Matching_Package\ppf_load_match.py"
call conda deactivate
pause