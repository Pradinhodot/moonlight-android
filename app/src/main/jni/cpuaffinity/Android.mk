# Builds libcpuaffinity.so — thread CPU pinning helpers (keep decode/render off the
# little cores to avoid scheduler-induced frame-time spikes). Picked up automatically
# by the parent jni/Android.mk via all-subdir-makefiles.
LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)
LOCAL_MODULE := cpuaffinity
LOCAL_SRC_FILES := cpuaffinity.cpp
LOCAL_CPPFLAGS := -std=c++14 -O2
LOCAL_LDLIBS := -llog
include $(BUILD_SHARED_LIBRARY)
