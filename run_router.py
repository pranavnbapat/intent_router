# run_router.py

import json
import sys

from router import route

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_router.py <your query...>")
        sys.exit(2)
    q = " ".join(sys.argv[1:])
    d = route(q)
    print(json.dumps(d.__dict__, indent=2))

if __name__ == "__main__":
    main()
