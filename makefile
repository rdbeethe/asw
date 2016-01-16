all:
	nvcc `pkg-config opencv --cflags` asw.cu `pkg-config opencv --libs`

run:
	optirun ./a.out l.png r.png 128 5 50
