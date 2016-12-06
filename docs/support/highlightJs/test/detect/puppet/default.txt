# EC2 sample

class ec2utils {

    # This must include the path to the Amazon EC2 tools
    $ec2path = ["/usr/bin", "/bin", "/usr/sbin", "/sbin",
                "/opt/ec2/ec2-api-tools/bin",
                "/opt/ec2/aws-elb-tools/bin"]

    define elasticip ($instanceid, $ip)
    {
        exec { "ec2-associate-address-$name":
            logoutput   => on_failure,
            environment => $ec2utils::ec2env,
            path        => $ec2utils::ec2path,
            command     => "ec2assocaddr $ip \
                                         -i $instanceid",
            # Only do this when necessary
            unless => "test `ec2daddr $ip | awk '{print \$3}'` == $instanceid",
        }
    }

    mount { "$mountpoint":
        device  => $devicetomount,
        ensure  => mounted,
        fstype  => $fstype,
        options => $mountoptions,
        require => [ Exec["ec2-attach-volume-$name"],
                     File["$mountpoint"]
        ],
    }

}
