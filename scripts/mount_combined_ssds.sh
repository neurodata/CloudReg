mkdir -p ~/ssd1 ~/ssd2
if [ $(($(lsblk | grep nvme | wc -l) - 2)) -lt $((4)) ]
then
    for blkdev in $(nvme list | awk '/^\/dev/ { print $1 }'); do
        i=1
        mapping=$(nvme id-ctrl --raw-binary "${blkdev}" | grep Instance)
        if [[ ${mapping} ]]; then
            echo "$blkdev is $mapping formatting and mounting..."
            mkfs.ext4 -E nodiscard -m0 ${blkdev}
            mount -o discard ${blkdev} /home/ubuntu/ssd{i}
            i++
        else
            echo "detected unknown drive letter $blkdev: $mapping. Skipping..."
        fi
    done
    chown ubuntu:ubuntu /home/ubuntu/ssd1
    chown ubuntu:ubuntu /home/ubuntu/ssdo2

else
    vgcreate LVMVolGroup 
    for blkdev in $(nvme list | awk '/^\/dev/ { print $1 }'); do
        i=0
        mapping=$(nvme id-ctrl --raw-binary "${blkdev}" | grep Instance)
        if [[ ${mapping} ]]; then
            echo "$blkdev is $mapping formatting and mounting..."
            pvcreate ${blkdev}
            if [[ ${i} == 0 ]]; then
                vgcreate LVMVolGroup ${blkdev}
            else
                echo ""
            fi
            vgextend LVMVolGroup ${blkdev}
            i++
        else
            echo "detected unknown drive letter $blkdev: $mapping. Skipping..."
        fi
    done
    lvcreate -l 50%FREE -n vol1 LVMVolGroup
    lvcreate -l 100%FREE -n vol2 LVMVolGroup
    #  create filesystems
    mkfs.ext4 -E nodiscard -m0 /dev/LVMVolGroup/vol1
    mkfs.ext4 -E nodiscard -m0 /dev/LVMVolGroup/vol2
    mount -o discard /dev/LVMVolGroup/vol1 /home/ubuntu/ssd1
    mount -o discard /dev/LVMVolGroup/vol2 /home/ubuntu/ssd2
    chown ubuntu:ubuntu -R /home/ubuntu/ssd1
    chown ubuntu:ubuntu -R /home/ubuntu/ssd2
fi
