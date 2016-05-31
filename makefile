CFLAGS=`pkg-config opencv --cflags`
LDFLAGS=`pkg-config opencv --libs` -L/usr/local/cuda-7.5/targets/x86_64/lib -lnpps -lnppi -lnppc

.PHONY: all
all: cost_volume asw

gpu_volume: cost_volume.cu
	nvcc $(CFLAGS) $^ -o $@ $(LDFLAGS)

cost_volume: cost_volume.o costVolumeFilter_jointBilateral.o costVolumeFilter_guided.o costVolumeFilter_box.o costVolumeMinimize.o createCostVolume_tadcg.o createCostVolume.o timer.o helper.o
	nvcc $^ -o $@ $(LDFLAGS)

%.o: %.cu %.h
	nvcc $(CFLAGS) -c $<

.PHONY: debug
debug: CFLAGS+= -g -G
debug: cost_volume

.PHONY: run
run: cost_volume
	./cost_volume

# the old implementation, still faster on some hardware
asw: asw.cu
	nvcc `pkg-config opencv --cflags` $< `pkg-config opencv --libs` -o $@
	# example for running the old version:
	# ./asw l.png r.png 64 5 50

.PHONY: clean
clean:
	rm asw cost_volume *.o

