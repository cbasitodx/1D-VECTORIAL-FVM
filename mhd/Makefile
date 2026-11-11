# MHD tests (CPU solver). Build them: make   /   Build and run them: make run
# The executables go to build/ (ignored by git). The tests write their results (.bin) in this folder: see cleanbin.
CXX      = g++
CXXFLAGS = -std=c++20 -Wall -Wextra -O3 -fno-math-errno -I../cpu -I../common

TESTS   = $(patsubst %.cpp,build/%,$(wildcard test*.cpp))
HEADERS = MHD.hpp $(wildcard ../common/include/*.h ../common/include/*.hpp ../cpu/include/*.hpp ../cpu/src/*.inl)

.PHONY: all run clean cleanbin

all: $(TESTS)

run: $(TESTS)
	for t in $(TESTS); do echo "== $$t"; ./$$t || exit 1; done

# Each test is one program: its .cpp and the binary writer. Rebuilt if the physics or the solver change
build/%: %.cpp ../common/src/HistoryBinary.cpp $(HEADERS)
	mkdir -p build
	$(CXX) $(CXXFLAGS) $< ../common/src/HistoryBinary.cpp -o $@ -lm

# Remove the build (the build/ folder)
clean:
	rm -rf build

# Remove the results written by the tests (in this folder)
cleanbin:
	rm -f *.bin
