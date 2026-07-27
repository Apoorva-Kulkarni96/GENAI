import asyncio

async def make_t():
    print("boil water")
    await asyncio.sleep(2)
    print("t : ready!")

async def make_t2():
    print("bake")
    await asyncio.sleep(1)
    print("t2: Ready!")


async def main():
    await asyncio.gather(make_t(), make_t2())

asyncio.run(main())