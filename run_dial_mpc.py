#!/usr/bin/env python3

import os
import sys

# Make sure we can import modules from tesi
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

if __name__ == "__main__":
    # Import here after adding to sys.path to avoid import errors
    from tesi.dial_mpc.core.dial_core import main
    
    # Run the main function from dial_core, which will parse its own arguments
    main() 