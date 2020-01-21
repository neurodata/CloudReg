findmnt /dev/nvme0n1
mkfs.ext4 -E nodiscard -m0 /dev/nvme0n1
mkfs.ext4 -E nodiscard -m0 /dev/nvme1n1
mount -o discard /dev/nvme0n1 /home/ubuntu/ssd1
mount -o discard /dev/nvme1n1 /home/ubuntu/ssd2
chown ubuntu:ubuntu /home/ubuntu/ssd1
chown ubuntu:ubuntu /home/ubuntu/ssd2
