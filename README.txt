usage: stator_reducer [-h] [--adjust LEFT RIGHT TOP BOTTOM] [--preserve PRESERVE]
                      [--forceblank FORCEBLANK] [--gun] [--shipchannel SHIPCHANNEL]
                      [--verbose VERBOSE] [--modeltest] [--inputformat INPUTFORMAT]
                      pattern period

A program to minimize the stator size of oscillators and guns in Conway's Game of Life

positional arguments:
  pattern               Filename of RLE containing object to be reduced
  period                Period of the object

options:
  -h, --help            show this help message and exit
  --adjust LEFT RIGHT TOP BOTTOM
                        Adjusts the size of the target bounding box to search. Must be
                        followed by four integers, indicating the change on the left,
                        right, top, and bottom of the bounding box. Negative numbers
                        decrease the size of the bounding box, positives increase it.
  --preserve PRESERVE   Filename of RLE where all cells containing in 1 will be preserved
                        in the reduced object. Should be the same size as the pattern.
                        Useful if you're trying to preserve clearance around a sparker.
  --forceblank FORCEBLANK
                        Filename of RLE where all cells containing the 1 mask stator cells
                        in the same original pattern, forcing them to be zero in the
                        modified oscillator. Useful for creating sparker variants with
                        additional clearance.
  --gun                 Processes the pattern as if it's a gun.
  --shipchannel SHIPCHANNEL
                        Filename of RLE which identifies all cells used by spaceships
                        created by the gun within the original bounding box. Should be the
                        same size as the pattern.
  --verbose VERBOSE     Set level of print verbosity: 0 is TEST (least), 2 is DEBUG
                        (most). Defaults to 1.
  --modeltest           Tests model against input pattern, which should always yield a
                        solution unless grid is being shrunk.
  --inputformat INPUTFORMAT
                        By default, the input file format for the oscillator is an RLE. However, other
                        formats are possible. Recognized values are ARRAY and KNOWNROTORS.

The primary difference between gun processing and oscillator processing is that for
oscillator processing, any cells in the pattern evolution that leave the original bounding
box are assumed to be part of the oscillator, and the original bounding box is expanded.
For gun processing, any cells in the pattern that leave the original bounding box are
assumed to be part of the ship channel.
