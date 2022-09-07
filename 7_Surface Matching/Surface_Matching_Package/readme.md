<h1>Open CV Surface Marching</h1>

<h2>Installation</h2>

download git data and move to directory in your cmd then enter
```
conda env  create -n OpenCVSurfaceMatching --file environment.yaml
```
Change path in [matching.bat]("https://github.com/MikeAlpaXRay/IFP_Hiwi/blob/master/7_Surface%20Matching/Surface_Matching_Package/matching.bat") and
in [parameters.py]("https://github.com/MikeAlpaXRay/IFP_Hiwi/blob/master/7_Surface%20Matching/Surface_Matching_Package/parameters.py") matching parameters and folders can be changed.


<h2>Usage</h2>
One Scene File and one or more Model Files with Prefix "ACTIVE_" or "AKTIVE_" are used for matching.

<h2>Cloudcreation</h2>
Provide a STI file of an object, [example files]("https://github.com/MikeAlpaXRay/IFP_Hiwi/tree/master/7_Surface%20Matching/Surface_Matching_Package/05_CAD").
Open set files with the program cloud compare, select the mesh and click on "Sample points on a mesh", keeping the normals.
These cloud can be exported as a non-binary PLY file.