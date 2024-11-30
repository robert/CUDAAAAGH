import asyncio
import struct
import time
from collections import defaultdict
from functools import wraps
from typing import ClassVar, Optional, Protocol

import httpx
import uvicorn
from aioconsole import ainput
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class Config:
    COMPUTE_ENDPOINT: str = "http://localhost:8000/compute"


def set_compute_endpoint(endpoint: str):
    Config.COMPUTE_ENDPOINT = endpoint


# Profiling setup
method_call_counts = defaultdict(int)
method_total_time = defaultdict(float)


def profile_calls(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()

        method_call_counts[func.__name__] += 1
        method_total_time[func.__name__] += end_time - start_time

        return result

    return wrapper


class IntOperator(Protocol):
    def compute(
        self, operation: str, left: int, right: Optional[int] = None
    ) -> int: ...


class LocalOperator:
    def compute(self, operation: str, left: int, right: Optional[int] = None) -> int:
        if operation == "add":
            return left + right
        elif operation == "sub":
            return left - right
        elif operation == "mul":
            return left * right
        elif operation == "floordiv":
            return left // right
        elif operation == "eq":
            return int(left == right)
        elif operation == "gt":
            return int(left > right)
        elif operation == "lt":
            return int(left < right)
        elif operation == "ge":
            return int(left >= right)
        elif operation == "le":
            return int(left <= right)
        elif operation == "xor":
            return left ^ right
        elif operation == "lshift":
            return left << right
        elif operation == "rshift":
            return left >> right
        elif operation == "and":
            return left & right
        elif operation == "or":
            return left | right
        elif operation == "invert":
            return ~left
        elif operation == "bit_length":
            if left == 0:
                return 1
            return left.bit_length()
        else:
            raise ValueError(f"Unknown operation: {operation}")


class RemoteOperator:
    def compute(self, operation: str, left: int, right: Optional[int] = None) -> int:
        request = {"operation": operation, "left": left, "right": right}

        response = httpx.post(Config.COMPUTE_ENDPOINT, json=request, timeout=60)
        if response.status_code != 200:
            raise RuntimeError(f"Remote operation failed: {response.text}")

        return response.json()["result"]


class ManualInt:
    operator: ClassVar[IntOperator]

    def __init__(self, val: int) -> None:
        if not isinstance(val, int):
            raise TypeError(f"{val=} not int")
        self.val: int = val

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.val})"

    def __index__(self) -> int:
        return self.val

    def __int__(self) -> int:
        return self.val

    @profile_calls
    def ith_bit(self, i: "ManualInt") -> "ManualInt":
        return self.__class__((self.val >> i.val) & 1)

    @profile_calls
    def __xor__(self, other: "ManualInt") -> "ManualInt":
        result = self.__class__(0)
        for i in range(max(self.bit_length(), other.bit_length())):
            bit_x = self.ith_bit(self.__class__(i))
            bit_y = other.ith_bit(self.__class__(i))
            xor_bit = bit_x + bit_y - self.__class__(2) * bit_x * bit_y
            result += xor_bit << self.__class__(i)
        return result

    @profile_calls
    def __floordiv__(self, other: "ManualInt") -> "ManualInt":
        if self < other:
            return self.__class__(0)
        ret = self.__class__(0)
        remaining = self
        while remaining >= other:
            remaining = remaining - other
            ret = ret + self.__class__(1)
        return ret

    @profile_calls
    def __lshift__(self, n: "ManualInt") -> "ManualInt":
        result = self
        for _ in range(n.val):
            result = result + result
        return result

    @profile_calls
    def __rshift__(self, n: "ManualInt") -> "ManualInt":
        """
        Implements right shift (>>) operation by repeatedly dividing by 2

        Similar to how left shift is implemented through repeated addition,
        right shift is implemented through repeated division by 2
        """
        result = self
        two = self.__class__(2)
        for _ in range(n.val):
            result = result // two
        return result

    @profile_calls
    def __and__(self, other: "ManualInt") -> "ManualInt":
        result = self.__class__(0)
        for i in range(max(self.bit_length(), other.bit_length())):
            bit_x = self.ith_bit(self.__class__(i))
            bit_y = other.ith_bit(self.__class__(i))
            and_bit = (
                self.__class__(1)
                if bit_x == self.__class__(1) and bit_y == self.__class__(1)
                else self.__class__(0)
            )
            result = result | (and_bit << self.__class__(i))
        return result

    @profile_calls
    def __or__(self, other: "ManualInt") -> "ManualInt":
        result = self.__class__(0)
        for i in range(max(self.bit_length(), other.bit_length())):
            bit_x = self.ith_bit(self.__class__(i))
            bit_y = other.ith_bit(self.__class__(i))
            or_bit = (
                self.__class__(1)
                if bit_x == self.__class__(1) or bit_y == self.__class__(1)
                else self.__class__(0)
            )
            result += or_bit << self.__class__(i)
        return result

    @profile_calls
    def bit_length(self) -> int:
        if self == self.__class__(0):
            return 1
        length = 0
        running = self
        while running > self.__class__(0):
            length += 1
            running = running // self.__class__(2)
        return length

    @profile_calls
    def __add__(self, other: "ManualInt") -> "ManualInt":
        return self.__class__(self.operator.compute("add", self.val, other.val))

    @profile_calls
    def __sub__(self, other: "ManualInt") -> "ManualInt":
        return self.__class__(self.operator.compute("sub", self.val, other.val))

    @profile_calls
    def __mul__(self, other: "ManualInt") -> "ManualInt":
        return self.__class__(self.operator.compute("mul", self.val, other.val))

    @profile_calls
    def __eq__(self, other: "ManualInt") -> bool:
        return bool(self.operator.compute("eq", self.val, other.val))

    @profile_calls
    def __gt__(self, other: "ManualInt") -> bool:
        return bool(self.operator.compute("gt", self.val, other.val))

    @profile_calls
    def __lt__(self, other: "ManualInt") -> bool:
        return bool(self.operator.compute("lt", self.val, other.val))

    @profile_calls
    def __ge__(self, other: "ManualInt") -> bool:
        return bool(self.operator.compute("ge", self.val, other.val))

    @profile_calls
    def __le__(self, other: "ManualInt") -> bool:
        return bool(self.operator.compute("le", self.val, other.val))


class ComputedInt(ManualInt):
    operator = LocalOperator()


class CUDAAAAGHInt(ManualInt):
    operator = RemoteOperator()


def rotate_right(value: ManualInt, shift: ManualInt) -> ManualInt:
    """Helper function for SHA-256 implementation"""
    return (value >> shift) | (
        value << (value.__class__(32) - shift)
    ) & value.__class__(0xFFFFFFFF)


def sha256(message: str, int_class: type[ManualInt] = ComputedInt) -> str:
    """
    Compute SHA-256 hash using specified ManualInt implementation

    Args:
        message: Input string to hash
        int_class: The ManualInt implementation to use (default: ComputedInt)
    """
    # Initialize hash values
    h0 = int_class(0x6A09E667)
    h1 = int_class(0xBB67AE85)
    h2 = int_class(0x3C6EF372)
    h3 = int_class(0xA54FF53A)
    h4 = int_class(0x510E527F)
    h5 = int_class(0x9B05688C)
    h6 = int_class(0x1F83D9AB)
    h7 = int_class(0x5BE0CD19)

    # Initialize round constants
    k = [
        int_class(x)
        for x in [
            0x428A2F98,
            0x71374491,
            0xB5C0FBCF,
            0xE9B5DBA5,
            0x3956C25B,
            0x59F111F1,
            0x923F82A4,
            0xAB1C5ED5,
            0xD807AA98,
            0x12835B01,
            0x243185BE,
            0x550C7DC3,
            0x72BE5D74,
            0x80DEB1FE,
            0x9BDC06A7,
            0xC19BF174,
            0xE49B69C1,
            0xEFBE4786,
            0x0FC19DC6,
            0x240CA1CC,
            0x2DE92C6F,
            0x4A7484AA,
            0x5CB0A9DC,
            0x76F988DA,
            0x983E5152,
            0xA831C66D,
            0xB00327C8,
            0xBF597FC7,
            0xC6E00BF3,
            0xD5A79147,
            0x06CA6351,
            0x14292967,
            0x27B70A85,
            0x2E1B2138,
            0x4D2C6DFC,
            0x53380D13,
            0x650A7354,
            0x766A0ABB,
            0x81C2C92E,
            0x92722C85,
            0xA2BFE8A1,
            0xA81A664B,
            0xC24B8B70,
            0xC76C51A3,
            0xD192E819,
            0xD6990624,
            0xF40E3585,
            0x106AA070,
            0x19A4C116,
            0x1E376C08,
            0x2748774C,
            0x34B0BCB5,
            0x391C0CB3,
            0x4ED8AA4A,
            0x5B9CCA4F,
            0x682E6FF3,
            0x748F82EE,
            0x78A5636F,
            0x84C87814,
            0x8CC70208,
            0x90BEFFFA,
            0xA4506CEB,
            0xBEF9A3F7,
            0xC67178F2,
        ]
    ]

    # Pre-processing
    message_bytes = bytearray(message, "utf-8")
    length = int_class(len(message_bytes) * 8)
    message_bytes.append(0x80)
    while (len(message_bytes) + 8) % 64 != 0:
        message_bytes.append(0x00)
    message_bytes += struct.pack(">Q", int(length))

    for i in range(0, len(message_bytes), 64):
        chunk = message_bytes[i : i + 64]
        w = [int_class(0)] * 64

        for j in range(16):
            w[j] = int_class(struct.unpack(">I", chunk[j * 4 : (j + 1) * 4])[0])

        for j in range(16, 64):
            s0 = (
                rotate_right(w[j - 15], int_class(7))
                ^ rotate_right(w[j - 15], int_class(18))
                ^ (w[j - 15] >> int_class(3))
            )
            s1 = (
                rotate_right(w[j - 2], int_class(17))
                ^ rotate_right(w[j - 2], int_class(19))
                ^ (w[j - 2] >> int_class(10))
            )
            w[j] = (w[j - 16] + s0 + w[j - 7] + s1) & int_class(0xFFFFFFFF)

        a, b, c, d = h0, h1, h2, h3
        e, f, g, h = h4, h5, h6, h7

        for j in range(64):
            S1 = (
                rotate_right(e, int_class(6))
                ^ rotate_right(e, int_class(11))
                ^ rotate_right(e, int_class(25))
            )
            ch = (e & f) ^ ((~e) & g)
            temp1 = (h + S1 + ch + k[j] + w[j]) & int_class(0xFFFFFFFF)
            S0 = (
                rotate_right(a, int_class(2))
                ^ rotate_right(a, int_class(13))
                ^ rotate_right(a, int_class(22))
            )
            maj = (a & b) ^ (a & c) ^ (b & c)
            temp2 = (S0 + maj) & int_class(0xFFFFFFFF)

            h = g
            g = f
            f = e
            e = (d + temp1) & int_class(0xFFFFFFFF)
            d = c
            c = b
            b = a
            a = (temp1 + temp2) & int_class(0xFFFFFFFF)

        h0 = (h0 + a) & int_class(0xFFFFFFFF)
        h1 = (h1 + b) & int_class(0xFFFFFFFF)
        h2 = (h2 + c) & int_class(0xFFFFFFFF)
        h3 = (h3 + d) & int_class(0xFFFFFFFF)
        h4 = (h4 + e) & int_class(0xFFFFFFFF)
        h5 = (h5 + f) & int_class(0xFFFFFFFF)
        h6 = (h6 + g) & int_class(0xFFFFFFFF)
        h7 = (h7 + h) & int_class(0xFFFFFFFF)

    return "{:08x}{:08x}{:08x}{:08x}{:08x}{:08x}{:08x}{:08x}".format(
        int(h0), int(h1), int(h2), int(h3), int(h4), int(h5), int(h6), int(h7)
    )


# --- SERVER ---


class Operation(BaseModel):
    operation: str
    left: int
    right: Optional[int] = None


class ManualComputationServer:
    def __init__(self, host="127.0.0.1", port=8000):
        self.host = host
        self.port = port
        self.app = FastAPI()
        self.results = {}

        # Register routes
        self.app.post("/compute")(self.compute)
        self.app.on_event("startup")(self.startup_event)

    def get_operation_description(self, op: Operation) -> str:
        """Convert operation into human-readable format"""
        if op.operation == "add":
            return f"{op.left} + {op.right}"
        elif op.operation == "sub":
            return f"{op.left} - {op.right}"
        elif op.operation == "mul":
            return f"{op.left} * {op.right}"
        elif op.operation == "floordiv":
            return f"{op.left} // {op.right}"
        elif op.operation == "eq":
            return f"{op.left} == {op.right}"
        elif op.operation == "gt":
            return f"{op.left} > {op.right}"
        elif op.operation == "lt":
            return f"{op.left} < {op.right}"
        elif op.operation == "ge":
            return f"{op.left} >= {op.right}"
        elif op.operation == "le":
            return f"{op.left} <= {op.right}"
        elif op.operation == "rshift":
            return f"{op.left} >> {op.right}"
        else:
            return f"{op.operation}({op.left}, {op.right})"

    async def input_handler(self):
        """Handle user input"""
        while True:
            try:
                result = await ainput()
                try:
                    # For boolean operations
                    if result.lower() in ("true", "false"):
                        result = 1 if result.lower() == "true" else 0
                    else:
                        result = int(result)

                    # Find the oldest waiting operation and complete it
                    waiting_ops = sorted(self.results.keys())
                    if waiting_ops:
                        op_id = waiting_ops[0]
                        self.results[op_id].set_result(result)
                except ValueError:
                    print("Invalid input. Please enter a number or true/false.")
            except Exception as e:
                print(f"Error handling input: {e}")

    async def compute(self, operation: Operation):
        op_id = id(operation)
        future = asyncio.Future()
        self.results[op_id] = future

        # Print the operation request
        desc = self.get_operation_description(operation)
        print(f"\nOperation requested: {desc}")
        print("Enter result: ", end="", flush=True)

        try:
            # Wait for the result
            result = await asyncio.wait_for(future, timeout=30.0)
            del self.results[op_id]  # Cleanup
            await asyncio.sleep(0.1)  # Ensure print order
            print(f"Returning result: {result}\n")
            return {"result": result}
        except asyncio.TimeoutError:
            del self.results[op_id]  # Cleanup
            raise HTTPException(
                status_code=408, detail="Operation timed out waiting for input"
            )
        except Exception as e:
            del self.results[op_id]  # Cleanup
            raise HTTPException(status_code=500, detail=str(e))

    async def startup_event(self):
        asyncio.create_task(self.input_handler())

    async def start(self):
        """Start the manual computation server asynchronously"""
        print("Starting manual computation server...")
        print("Operations will be displayed here for manual computation.")
        print("Enter the results when prompted.")
        config = uvicorn.Config(self.app, host=self.host, port=self.port)
        server = uvicorn.Server(config)
        await server.serve()


async def start_server(host="127.0.0.1", port=8000):
    """Helper function to quickly start a server with async support"""
    server = ManualComputationServer(host=host, port=port)
    await server.start()
