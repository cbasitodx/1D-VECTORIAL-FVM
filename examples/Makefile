# Build the tests of one version:          make cpu      /  make cuda
# Build and run the tests of one version:  make runcpu   /  make runcuda
#
# Everything the build produces goes to build/ (ignored by git), so the source folders stay clean:
#   build/obj/   object files (.o), one per test plus HistoryBinary.o
#   build/       the test executables, e.g. build/testBurgers
# The tests write their results (.bin) in this folder, examples/: see cleanbin.
CXX       = g++
NVCC       = nvcc
# C++20 is needed for the PhysicsType concept (common/include/Physics.hpp). nvcc supports it from CUDA 12
CXXFLAGS   = -std=c++20 -Wall -Wextra -I../cpu -I../common
NVCCFLAGS  = -std=c++20 -I../cuda -I../common -arch=sm_75
# Optimization flags, kept apart so that they can be changed on their own (benchmark.sh does).
# -fno-math-errno: sqrt does not need to report errors, which lets the compiler optimize loops that use it
OPTFLAGS   = -O3 -fno-math-errno
NVOPTFLAGS = -O3
LIBS       = -lm

# Folders where make looks for source files. Each example is a folder with its physics and its tests
# (the solvers in ../cpu/src/*.inl and ../cuda/src/*.inl are not compiled on their own: they are included through their headers)
EXAMPLES = linear_transport burgers shallow_water euler
vpath %.cpp $(EXAMPLES)    # CPU test programs
vpath %.cu  $(EXAMPLES)    # CUDA test programs
vpath %.cpp ../common/src  # shared code (HistoryBinary.cpp)

# One executable per test (burgers/testBurgers.cpp -> build/testBurgers, burgers/testBurgersCUDA.cu -> build/testBurgersCUDA)
CPU_TESTS    = $(patsubst %.cpp,build/%,$(notdir $(wildcard $(addsuffix /test*.cpp,$(EXAMPLES)))))
CUDA_TESTS   = $(patsubst %.cu,build/%,$(notdir $(wildcard $(addsuffix /test*.cu,$(EXAMPLES)))))
COMMON       = build/obj/HistoryBinary.o
HEADERS      = $(wildcard */*.hpp ../common/include/*.h ../common/include/*.hpp)
CPU_HEADERS  = $(HEADERS) $(wildcard ../cpu/include/*.hpp ../cpu/src/*.inl)
CUDA_HEADERS = $(HEADERS) $(wildcard ../cuda/include/*.cuh ../cuda/src/*.inl)

.PHONY: cpu cuda runcpu runcuda clean cleanbin

cpu: $(CPU_TESTS)

cuda: $(CUDA_TESTS)

# Build and run every test. Output files are written in this folder
runcpu: $(CPU_TESTS)
	for t in $(CPU_TESTS); do echo "== $$t"; ./$$t || exit 1; done

runcuda: $(CUDA_TESTS)
	for t in $(CUDA_TESTS); do echo "== $$t"; ./$$t || exit 1; done

# Link: each test = its own object + the shared objects
$(CPU_TESTS): build/%: build/obj/%.o $(COMMON)
	$(CXX) $^ -o $@ $(LIBS)

$(CUDA_TESTS): build/%: build/obj/%.o $(COMMON)
	$(NVCC) $^ -o $@ $(LIBS)

# Compile: any X.cpp (found through vpath) -> build/obj/X.o with g++. Recompile if a header or the solver changes
build/obj/%.o: %.cpp $(CPU_HEADERS)
	mkdir -p build/obj
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -c $< -o $@

# Compile: any X.cu (found through vpath) -> build/obj/X.o with nvcc. Recompile if a header or the solver changes
build/obj/%.o: %.cu $(CUDA_HEADERS)
	mkdir -p build/obj
	$(NVCC) $(NVCCFLAGS) $(NVOPTFLAGS) -c $< -o $@

# Remove everything the build produced (the build/ folder)
clean:
	rm -rf build

# Remove the binary history files written by the tests (in this folder)
cleanbin:
	rm -f *.bin
