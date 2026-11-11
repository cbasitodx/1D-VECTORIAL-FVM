# Build and run every test of one version:  make cpu  /  make cuda
# Build and run the MHD tests (mhd/):       make mhd
.PHONY: cpu cuda mhd clean

cpu:
	$(MAKE) -C examples runcpu

cuda:
	$(MAKE) -C examples runcuda

mhd:
	$(MAKE) -C mhd run

clean:
	$(MAKE) -C examples clean
