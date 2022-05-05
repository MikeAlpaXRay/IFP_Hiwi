<h1>Open CV Surface Marching</h1>

Tool to convert the CSV-File form the <a href="https://r6analyst.com/">R6 ANALYST</a> to compact fileformat.
<br>
Goal to enable datasience not based on Excel
<h2>Installation</h2>

download git data and move to directory in your cmd then enter
```
conda env  create -n OpenCVSurfaceMatching --file environment.yaml
```
Change path in [matching.bat](Link) and if desired DEFAULT user, respectively.
In [parameters.py](Link) matching parameters and folders can be changed.



<h1>r6stats</h1>
Date: 2021.03.06 <br>
Tool to convert the CSV-File form the <a href="https://r6analyst.com/">R6 ANALYST</a> to compact fileformat.
<br>
Goal to enable datasience not based on Excel
<h2>Installation</h2>

download git data and move to tool directory in your cmd then enter
```
mkdir data
conda env create -n R6_datatool --file environment.yaml python=3.8.8
```
Change path in [addData.bat](https://github.com/MikeAlpaXRay/R6_datatool/blob/main/addData.bat) and if desired DEFAULT user, respectively.
In [user_constants.py](https://github.com/MikeAlpaXRay/R6_datatool/blob/main/user_constants.py) if desired change values of constants.
<br>

<h2>Maintenance</h2>
In [program_constants.py](https://github.com/MikeAlpaXRay/R6_datatool/blob/main/program_constants.py) it is possible to add new operators or change map pools.

<h2>Usage</h2>
One Scene File and one or more Model Files with Prefix "ACTIVE_" or "AKTIVE_" are used for matching.