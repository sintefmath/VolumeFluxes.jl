# Copyright (c) 2024 SINTEF AS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

using Parameters

"""
	IntervalWriter{WriterType}

A structure for writing data at specified intervals using a user
specified writer of type `WriterType`. This avoids having every
timestep written.

# Type Parameters

- `WriterType`: The type of the writer used for data output.

# Description

`IntervalWriter` is designed to periodically write data during simulations.
It utilizes a writer of the specified `WriterType` to output data at defined 
intervals.
"""
@with_kw mutable struct IntervalWriter{WriterType}
	current_t::Float64 = 0.0
	step::Float64
	writer::WriterType
end

IntervalWriter(step::Real, writer::Base.Callable) = IntervalWriter(step=step, writer=writer)
IntervalWriter(writer::Base.Callable) = IntervalWriter(step=1.0, writer=writer)

function (writer::IntervalWriter)(t, simulator)
	dt = VolumeFluxes.current_timestep(simulator)
	if t + dt >= writer.current_t
		writer.writer(t, simulator)
		writer.current_t += writer.step
	end
end

"""
    MultipleCallbacks(callbacks)

A callback aggregator that allows multiple callbacks to be executed sequentially.

# Arguments
- `callbacks`: Collection of callback functions to be executed

# Fields
- `callbacks`: Stored collection of callback functions

Each callback in the collection should be a function that accepts two arguments:
- `t`: The current time
- `simulator`: The simulator state/object

# Example
"""
struct MultipleCallbacks
	callbacks::Any
end

function (mc::MultipleCallbacks)(t, simulator)
	for c in mc.callbacks
		c(t, simulator)
	end
end
