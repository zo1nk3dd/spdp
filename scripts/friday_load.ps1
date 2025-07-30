# build and load the executable into the hpc
# windows script

cargo build --release

$path = "C:\Users\zthom\Documents\Projects\spdp\target\release\spdp.exe"
$destination =  "username@friday.research.dc.uq.edu.au:/executables/"

scp $path $destination
