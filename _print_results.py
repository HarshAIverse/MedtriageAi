"""
_print_results.py - Prints all 46 lines of val_log.txt as JSON array to avoid terminal truncation
"""
import json
lines = open("val_log.txt", encoding="ascii", errors="replace").readlines()
print(json.dumps([l.rstrip("\n") for l in lines], indent=2))
