from CUDAAAAGH import CUDAAAAGHInt, set_compute_endpoint, sha256


def main():
    # Configure the endpoint where our server is running
    set_compute_endpoint("http://localhost:8000/compute")

    # Basic arithmetic operations
    print("\n=== Basic Arithmetic ===")
    a = CUDAAAAGHInt(10)
    b = CUDAAAAGHInt(5)

    print(f"\nComputing {a} + {b}...")
    sum_result = a + b
    print(f"Sum result: {sum_result}")

    print(f"\nComputing {a} * {b}...")
    mul_result = a * b
    print(f"Multiplication result: {mul_result}")

    print(f"\nComputing {a} - {b}...")
    mul_result = a - b
    print(f"Subtraction result: {mul_result}")

    print(f"\nComputing {a} / {b}...")
    mul_result = a // b
    print(f"Division result: {mul_result}")

    # Bitwise operations
    print("\n=== Bitwise Operations ===")
    x = CUDAAAAGHInt(12)  # 1100 in binary
    y = CUDAAAAGHInt(10)  # 1010 in binary

    print(f"\nComputing {x} & {y} (bitwise AND)...")
    and_result = x & y
    print(f"AND result: {and_result}")

    print(f"\nComputing {x} | {y} (bitwise OR)...")
    or_result = x | y
    print(f"OR result: {or_result}")

    # SHA-256 example (warning: this will require many, many, many operations)
    print("\n=== SHA-256 Hash ===")
    message = "Hello!"
    print(f"Computing SHA-256 hash of '{message}'...")
    hash_result = sha256(message, CUDAAAAGHInt)
    print(f"Hash result: {hash_result}")


if __name__ == "__main__":
    main()
