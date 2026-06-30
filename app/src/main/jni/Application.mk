# Application.mk for Moonlight

# Our minimum version is Android 5.0
APP_PLATFORM := android-21

# We support 16KB pages
APP_SUPPORT_FLEXIBLE_PAGE_SIZES := true

# libcpuaffinity.cpp uses the C++ STL (vector/string/fstream). The rest of the
# native code is C, and cpuaffinity is a self-contained JNI lib with no STL
# objects crossing library boundaries, so a private static libc++ is safe here.
APP_STL := c++_static
