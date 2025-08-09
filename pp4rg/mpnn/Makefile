PYTHON_VERSION=$(shell python --version | cut -d' ' -f2 | cut -d. -f1,2)

CXX=g++
CXXFLAGS=-O3 -Wall -shared -std=c++11 -fPIC
PYBIND11_FLAGS=$(shell python$(PYTHON_VERSION) -m pybind11 --includes)

SRC=$(wildcard *.cc)

OUTPUT=mpnn$(shell python$(PYTHON_VERSION)-config --extension-suffix)

all: $(OUTPUT)

$(OUTPUT): $(SRC)
	$(CXX) $(CXXFLAGS) $(PYBIND11_FLAGS) $(SRC) -o $@

clean:
	rm -f $(OBJS) $(OUTPUT)

