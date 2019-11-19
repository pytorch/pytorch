package org.pytorch.android;

import org.pytorch.ReadAdapter;

import java.io.Closeable;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import android.content.res.AssetManager;
import android.content.res.AssetFileDescriptor;

public class AssetReadAdapter implements ReadAdapter, Closeable {
    private final long size;
    private final long startOffset;
    private final FileInputStream fileInputStream;
    private final FileChannel fileChannel;

    public AssetReadAdapter(AssetManager assetManager, String assetName) throws IOException {
        AssetFileDescriptor fd = assetManager.openFd(assetName);
        this.size = fd.getLength();
        this.startOffset = fd.getStartOffset();
        this.fileInputStream = fd.createInputStream();
        this.fileChannel = fileInputStream.getChannel();
    }

    @Override
    public long size() {
        return size;
    }

    @Override
    public int read(long position, ByteBuffer buffer, int size) throws IOException {
        return fileChannel.read(buffer, startOffset + position);
    }

    @Override
    public void close() throws IOException {
        fileInputStream.close();
    }
}
