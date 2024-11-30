import asyncio

from CUDAAAAGH import start_server

if __name__ == "__main__":
    asyncio.run(start_server(host="127.0.0.1", port=8000))
