// multiview.slang.glsl
#version 450
#extension GL_EXT_multiview : require

void main()
{
    // Cast to uint as the GLSL extension types gl_ViewIndex as `highp int`
    gl_Position = vec4(float(uint(gl_ViewIndex)), 0, 0, 0);
    return;
}

