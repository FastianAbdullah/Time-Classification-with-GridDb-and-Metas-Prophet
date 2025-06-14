import sys
import os

# Add GridDB Python client path (check your actual version)
griddb_path = r"C:\Program Files\GridDB\Python Client\5.5.0"  # Adjust version number
sys.path.insert(0, griddb_path)

import griddb_python as griddb

try:
    # Connect to your GridDB Docker container
    factory = griddb.StoreFactory.get_instance()
    store = factory.get_store(
        notification_member="127.0.0.1:10001",
        cluster_name="defaultCluster",
        username="admin",
        password="admin"
    )
    print("Successfully connected to GridDB!")
    
    # Test with a simple container operation
    container_name = "test_container"
    
    # Create container info
    container_info = griddb.ContainerInfo(container_name,
        [["id", griddb.Type.INTEGER],
         ["name", griddb.Type.STRING],
         ["value", griddb.Type.DOUBLE]],
        griddb.ContainerType.COLLECTION, True)
    
    # Get or create container
    container = store.put_container(container_info)
    print(f"Container '{container_name}' created/accessed successfully!")
    
except Exception as e:
    print(f"Error: {e}")
    print("Make sure GridDB Docker container is running and accessible")