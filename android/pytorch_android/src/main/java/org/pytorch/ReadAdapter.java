package org.pytorch;

import java.io.IOException;
import java.nio.ByteBuffer;

public interface ReadAdapter {
    long size();
    int read(long position, ByteBuffer buffer, int size) throws IOException;
}
