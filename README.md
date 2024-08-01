# WDSLAM_CPP
SLAM Implementation using C++ and 




To resolve the situation

/usr/bin/ld: /home/wondong/anaconda3/lib/libgdk_pixbuf-2.0.so.0: undefined reference to `g_task_set_static_name'

-> The reason is the environment varialbe error in Anaconda3

``` bash
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/lib:$LD_LIBRARY_PATH
```