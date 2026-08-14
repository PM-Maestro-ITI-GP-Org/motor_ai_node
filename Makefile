# motor_ai_node links nothing -- no CommonAPI, no vsomeip, no boost, not even a
# math library -- so there is no configure step, no toolchain file and no
# dependency on the SOME/IP tree that motor_ai_server needs. One compiler and
# one translation unit.
#
# CXX and CXXFLAGS are ?= so the environment or the command line can replace
# them, which is how Yocto cross-compiles this:
#
#     make CXX="$CXX" CXXFLAGS="$CXXFLAGS -std=c++14" LDFLAGS="$LDFLAGS"
#
# LDFLAGS is in the rule because the compile and the link are one invocation;
# OE's QA check fails a package whose binaries were linked without them.
CXX      ?= g++
CXXFLAGS ?= -O2 -Wall -Wextra -std=c++14

BIN  := motor_ai_node
SRCS := src/motor_ai_node.cpp

.PHONY: all clean

all: $(BIN)

$(BIN): $(SRCS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $<

clean:
	rm -f $(BIN)
