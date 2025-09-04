import asyncio
from bleak import BleakScanner, BleakClient

# Replace this once you find Inkbird UUID from scan
ADDRESS = None  # e.g., "D4D6A5E5-12B1-4A5D-BE7E-XXXXXXXXXXXX"
NOTIFY_CHAR_UUID = "5833ff03-9b8b-5191-6142-22a4536ef123"  # replace with your Notify characteristic

async def scan_devices():
    print("🔍 Scanning for nearby BLE devices...")
    devices = await BleakScanner.discover()
    for d in devices:
        print(f"Name: {d.name}, ID: {d.address}")  # ID is the UUID you need on macOS
    print("\n➡️ Find your Inkbird in the list above and copy its ID into ADDRESS.")

async def connect_and_listen(address: str):
    async with BleakClient(address) as client:
        print(f"✅ Connected to {address}")

        def handle_notify(sender, data: bytearray):
            print(f"📡 Notification from {sender}: {list(data)}")

        await client.start_notify(NOTIFY_CHAR_UUID, handle_notify)
        print("📥 Listening for temperature data... (press Ctrl+C to stop)")
        await asyncio.sleep(60)
        await client.stop_notify(NOTIFY_CHAR_UUID)

async def main():
    if ADDRESS is None:
        await scan_devices()
    else:
        await connect_and_listen(ADDRESS)

if __name__ == "__main__":
    asyncio.run(main())
