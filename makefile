all: cpu_asw

gpu_asw: asw.cu
	nvcc `pkg-config opencv --cflags` $^ -o $@ `pkg-config opencv --libs`
	# optirun ./a.out l.png r.png 64 5 50

cpu_asw: asw.cpp
	g++ `pkg-config opencv --cflags` -pthread $^ -o $@ `pkg-config opencv --libs` -pthread

run: cpu_asw
	./cpu_asw
