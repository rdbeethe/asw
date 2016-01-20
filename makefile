all:
	nvcc --ptx --ptxas-options=-v `pkg-config opencv --cflags` asw.cu `pkg-config opencv --libs`

run:
	optirun ./a.out l.png r.png 64 5 50
