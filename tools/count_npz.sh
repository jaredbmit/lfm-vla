#!/bin/bash
find "${1:-.}" -name "*.npz" | wc -l
