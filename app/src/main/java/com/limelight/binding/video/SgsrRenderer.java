package com.limelight.binding.video;

import android.graphics.SurfaceTexture;
import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
import android.view.Surface;

import com.limelight.LimeLog;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.concurrent.atomic.AtomicBoolean;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

/**
 * On-device GPU upscaler for the 2D stream path: a two-pass FSR1-class pipeline.
 *
 *   Pass 1 (EASU): edge-adaptive spatial upscale. For each output pixel it gathers a 4x4
 *                  source neighbourhood, estimates the local edge direction from the luma
 *                  gradient, and applies an anisotropic Lanczos-2 kernel that is TIGHT across
 *                  the edge (keeps it crisp) and LOOSE along it (no staircasing). Samples the
 *                  decoder's external-OES texture, renders into an intermediate FBO at panel res.
 *   Pass 2 (RCAS): robust contrast-adaptive sharpening with a neighbourhood clamp (no ringing),
 *                  reads the FBO, renders to the screen.
 *
 * Stream low -> tiny frames (locked fps, no thermal throttle, no Wi-Fi microbursts), reconstruct
 * sharpness on the Adreno. Temporal upscalers (FSR2/XeSS/TSR) are impossible here: a video stream
 * has no motion vectors, no depth and no sub-pixel jitter, so spatial (FSR1) is the quality ceiling.
 *
 * Modelled on the proven GL plumbing in {@code Stereo3DRenderer}. Behind a toggle (default OFF);
 * when disabled the normal direct-to-SurfaceView path is used and this class is never touched.
 * Falls back gracefully (EASU-straight-to-screen, then passthrough) if the FBO/programs fail.
 */
public class SgsrRenderer implements GLSurfaceView.Renderer, SurfaceTexture.OnFrameAvailableListener {

    public interface OnSgsrSurfaceReadyListener {
        void onSgsrSurfaceReady(Surface surface);
    }

    private static final int GL_TEXTURE_EXTERNAL_OES = 0x8D65;

    private static final float[] QUAD_VERTICES = {-1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f};
    // OES output is Y-flipped, so pass 1 (sampling the OES texture) uses flipped tex coords...
    private static final float[] TEX_OES = {0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f};
    // ...and pass 2 (sampling the upright FBO 2D texture) uses normal tex coords.
    private static final float[] TEX_2D = {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};

    private static final String VERTEX_SHADER =
            "attribute vec4 a_Position;\n" +
            "attribute vec2 a_TexCoord;\n" +
            "varying vec2 v_TexCoord;\n" +
            "void main() {\n" +
            "  gl_Position = a_Position;\n" +
            "  v_TexCoord = a_TexCoord;\n" +
            "}\n";

    // EASU: edge-adaptive 4x4 anisotropic Lanczos-2 upscale, sampling the external-OES texture.
    private static final String EASU_FRAGMENT =
            "#extension GL_OES_EGL_image_external : require\n" +
            "precision highp float;\n" +
            "varying vec2 v_TexCoord;\n" +
            "uniform samplerExternalOES s_Texture;\n" +
            "uniform vec2 u_inRes;\n" +
            "vec3 fetch(vec2 ip){ return texture2D(s_Texture, (ip + 0.5) / u_inRes).rgb; }\n" +
            "float luma(vec3 c){ return dot(c, vec3(0.299, 0.587, 0.114)); }\n" +
            "float sinc(float x){ x = max(abs(x), 1e-4) * 3.14159265; return sin(x) / x; }\n" +
            "float lanczos2(float x){ x = abs(x); if (x >= 2.0) return 0.0; return sinc(x) * sinc(x * 0.5); }\n" +
            "void tap(float ox, float oy, vec2 fp, vec2 pf, vec2 edge, float aA, float aL, inout vec3 acc, inout float ws){\n" +
            "  vec3 c = fetch(fp + vec2(ox, oy));\n" +
            "  vec2 o = vec2(ox, oy) - pf;\n" +
            "  float al = dot(o, edge);\n" +
            "  float ac = dot(o, vec2(-edge.y, edge.x));\n" +
            "  float r = length(vec2(ac * aA, al * aL));\n" +
            "  float w = lanczos2(r);\n" +
            "  acc += c * w; ws += w;\n" +
            "}\n" +
            "void main(){\n" +
            "  vec2 pp = v_TexCoord * u_inRes - 0.5;\n" +
            "  vec2 fp = floor(pp);\n" +
            "  vec2 pf = pp - fp;\n" +
            "  float l00 = luma(fetch(fp + vec2(0.0, 0.0)));\n" +
            "  float l10 = luma(fetch(fp + vec2(1.0, 0.0)));\n" +
            "  float l01 = luma(fetch(fp + vec2(0.0, 1.0)));\n" +
            "  float l11 = luma(fetch(fp + vec2(1.0, 1.0)));\n" +
            "  float gx = (l10 + l11) - (l00 + l01);\n" +
            "  float gy = (l01 + l11) - (l00 + l10);\n" +
            "  vec2 grad = vec2(gx, gy);\n" +
            "  float gmag = length(grad);\n" +
            "  float strength = clamp(gmag * 4.0, 0.0, 1.0);\n" +
            "  vec2 edge = (gmag > 1e-4) ? normalize(vec2(-grad.y, grad.x)) : vec2(1.0, 0.0);\n" +
            "  float aAcross = mix(1.0, 1.6, strength);\n" +
            "  float aAlong  = mix(1.0, 0.7, strength);\n" +
            "  vec3 acc = vec3(0.0); float ws = 0.0;\n" +
            "  tap(-1.0,-1.0,fp,pf,edge,aAcross,aAlong,acc,ws); tap(0.0,-1.0,fp,pf,edge,aAcross,aAlong,acc,ws);\n" +
            "  tap( 1.0,-1.0,fp,pf,edge,aAcross,aAlong,acc,ws); tap(2.0,-1.0,fp,pf,edge,aAcross,aAlong,acc,ws);\n" +
            "  tap(-1.0, 0.0,fp,pf,edge,aAcross,aAlong,acc,ws); tap(0.0, 0.0,fp,pf,edge,aAcross,aAlong,acc,ws);\n" +
            "  tap( 1.0, 0.0,fp,pf,edge,aAcross,aAlong,acc,ws); tap(2.0, 0.0,fp,pf,edge,aAcross,aAlong,acc,ws);\n" +
            "  tap(-1.0, 1.0,fp,pf,edge,aAcross,aAlong,acc,ws); tap(0.0, 1.0,fp,pf,edge,aAcross,aAlong,acc,ws);\n" +
            "  tap( 1.0, 1.0,fp,pf,edge,aAcross,aAlong,acc,ws); tap(2.0, 1.0,fp,pf,edge,aAcross,aAlong,acc,ws);\n" +
            "  tap(-1.0, 2.0,fp,pf,edge,aAcross,aAlong,acc,ws); tap(0.0, 2.0,fp,pf,edge,aAcross,aAlong,acc,ws);\n" +
            "  tap( 1.0, 2.0,fp,pf,edge,aAcross,aAlong,acc,ws); tap(2.0, 2.0,fp,pf,edge,aAcross,aAlong,acc,ws);\n" +
            "  vec3 outc = (ws > 0.0) ? acc / ws : fetch(fp + vec2(0.0, 0.0));\n" +
            "  gl_FragColor = vec4(clamp(outc, 0.0, 1.0), 1.0);\n" +
            "}\n";

    // RCAS: contrast-adaptive sharpen with neighbourhood clamp (no ringing), on the upscaled image.
    private static final String RCAS_FRAGMENT =
            "precision highp float;\n" +
            "varying vec2 v_TexCoord;\n" +
            "uniform sampler2D s_Texture;\n" +
            "uniform vec2 u_texelSize;\n" +
            "uniform float u_sharpness;\n" +
            "void main(){\n" +
            "  vec3 c = texture2D(s_Texture, v_TexCoord).rgb;\n" +
            "  vec3 n = texture2D(s_Texture, v_TexCoord + vec2(0.0, -u_texelSize.y)).rgb;\n" +
            "  vec3 s = texture2D(s_Texture, v_TexCoord + vec2(0.0,  u_texelSize.y)).rgb;\n" +
            "  vec3 w = texture2D(s_Texture, v_TexCoord + vec2(-u_texelSize.x, 0.0)).rgb;\n" +
            "  vec3 e = texture2D(s_Texture, v_TexCoord + vec2( u_texelSize.x, 0.0)).rgb;\n" +
            "  vec3 blur = (n + s + w + e) * 0.25;\n" +
            "  vec3 sharp = c + (c - blur) * (u_sharpness * 2.0);\n" +
            "  vec3 lo = min(c, min(min(n, s), min(w, e)));\n" +
            "  vec3 hi = max(c, max(max(n, s), max(w, e)));\n" +
            "  gl_FragColor = vec4(clamp(sharp, lo, hi), 1.0);\n" +
            "}\n";

    private final GLSurfaceView glSurfaceView;
    private final OnSgsrSurfaceReadyListener listener;
    private final int inputWidth;
    private final int inputHeight;
    private final float sharpness;

    private final FloatBuffer quadBuffer;
    private final FloatBuffer texOesBuffer;
    private final FloatBuffer tex2dBuffer;
    private final AtomicBoolean frameAvailable = new AtomicBoolean(false);

    private int easuProgram;
    private int easuPos, easuTex, easuSampler, easuInRes;
    private int rcasProgram;
    private int rcasPos, rcasTex, rcasSampler, rcasTexel, rcasSharp;

    private int oesTextureId;
    private SurfaceTexture videoSurfaceTexture;
    private Surface videoSurface;

    private int fboId;
    private int fboTexId;
    private int outWidth;
    private int outHeight;

    public SgsrRenderer(GLSurfaceView view, OnSgsrSurfaceReadyListener listener,
                        int inputWidth, int inputHeight, float sharpness) {
        this.glSurfaceView = view;
        this.listener = listener;
        this.inputWidth = Math.max(1, inputWidth);
        this.inputHeight = Math.max(1, inputHeight);
        this.sharpness = sharpness;

        quadBuffer = floatBuffer(QUAD_VERTICES);
        texOesBuffer = floatBuffer(TEX_OES);
        tex2dBuffer = floatBuffer(TEX_2D);
    }

    private static FloatBuffer floatBuffer(float[] data) {
        FloatBuffer b = ByteBuffer.allocateDirect(data.length * 4)
                .order(ByteOrder.nativeOrder()).asFloatBuffer();
        b.put(data).position(0);
        return b;
    }

    public Surface getVideoSurface() {
        return videoSurface;
    }

    @Override
    public void onFrameAvailable(SurfaceTexture surfaceTexture) {
        frameAvailable.set(true);
        glSurfaceView.requestRender();
    }

    @Override
    public void onSurfaceCreated(GL10 gl, EGLConfig config) {
        oesTextureId = createExternalOESTexture();
        videoSurfaceTexture = new SurfaceTexture(oesTextureId);
        videoSurfaceTexture.setOnFrameAvailableListener(this);
        videoSurface = new Surface(videoSurfaceTexture);

        easuProgram = createProgram(VERTEX_SHADER, EASU_FRAGMENT);
        if (easuProgram != 0) {
            easuPos = GLES20.glGetAttribLocation(easuProgram, "a_Position");
            easuTex = GLES20.glGetAttribLocation(easuProgram, "a_TexCoord");
            easuSampler = GLES20.glGetUniformLocation(easuProgram, "s_Texture");
            easuInRes = GLES20.glGetUniformLocation(easuProgram, "u_inRes");
        }
        rcasProgram = createProgram(VERTEX_SHADER, RCAS_FRAGMENT);
        if (rcasProgram != 0) {
            rcasPos = GLES20.glGetAttribLocation(rcasProgram, "a_Position");
            rcasTex = GLES20.glGetAttribLocation(rcasProgram, "a_TexCoord");
            rcasSampler = GLES20.glGetUniformLocation(rcasProgram, "s_Texture");
            rcasTexel = GLES20.glGetUniformLocation(rcasProgram, "u_texelSize");
            rcasSharp = GLES20.glGetUniformLocation(rcasProgram, "u_sharpness");
        }
        LimeLog.info("SGSR (FSR1: EASU+RCAS) ready, input " + inputWidth + "x" + inputHeight
                + " easu=" + (easuProgram != 0) + " rcas=" + (rcasProgram != 0));

        if (listener != null) {
            listener.onSgsrSurfaceReady(videoSurface);
        }
    }

    @Override
    public void onSurfaceChanged(GL10 gl, int width, int height) {
        outWidth = width;
        outHeight = height;
        GLES20.glViewport(0, 0, width, height);
        createFbo(width, height);
    }

    @Override
    public void onDrawFrame(GL10 gl) {
        if (!frameAvailable.getAndSet(false)) {
            return;
        }
        try {
            videoSurfaceTexture.updateTexImage();
        } catch (Exception e) {
            LimeLog.warning("SGSR updateTexImage failed: " + e.getMessage());
            return;
        }

        if (easuProgram == 0) {
            GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT);
            return;
        }

        boolean twoPass = (fboId != 0 && fboTexId != 0 && rcasProgram != 0);

        // --- Pass 1: EASU (OES -> FBO if two-pass, else straight to screen) ---
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, twoPass ? fboId : 0);
        GLES20.glViewport(0, 0, outWidth, outHeight);
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT);
        GLES20.glUseProgram(easuProgram);
        GLES20.glVertexAttribPointer(easuPos, 2, GLES20.GL_FLOAT, false, 0, quadBuffer);
        GLES20.glVertexAttribPointer(easuTex, 2, GLES20.GL_FLOAT, false, 0, texOesBuffer);
        GLES20.glEnableVertexAttribArray(easuPos);
        GLES20.glEnableVertexAttribArray(easuTex);
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GL_TEXTURE_EXTERNAL_OES, oesTextureId);
        GLES20.glUniform1i(easuSampler, 0);
        GLES20.glUniform2f(easuInRes, inputWidth, inputHeight);
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);
        GLES20.glDisableVertexAttribArray(easuPos);
        GLES20.glDisableVertexAttribArray(easuTex);

        if (!twoPass) {
            return;
        }

        // --- Pass 2: RCAS (FBO -> screen) ---
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);
        GLES20.glViewport(0, 0, outWidth, outHeight);
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT);
        GLES20.glUseProgram(rcasProgram);
        GLES20.glVertexAttribPointer(rcasPos, 2, GLES20.GL_FLOAT, false, 0, quadBuffer);
        GLES20.glVertexAttribPointer(rcasTex, 2, GLES20.GL_FLOAT, false, 0, tex2dBuffer);
        GLES20.glEnableVertexAttribArray(rcasPos);
        GLES20.glEnableVertexAttribArray(rcasTex);
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, fboTexId);
        GLES20.glUniform1i(rcasSampler, 0);
        GLES20.glUniform2f(rcasTexel, 1.0f / outWidth, 1.0f / outHeight);
        GLES20.glUniform1f(rcasSharp, sharpness);
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);
        GLES20.glDisableVertexAttribArray(rcasPos);
        GLES20.glDisableVertexAttribArray(rcasTex);
    }

    public void release() {
        if (videoSurface != null) {
            videoSurface.release();
            videoSurface = null;
        }
        if (videoSurfaceTexture != null) {
            videoSurfaceTexture.release();
            videoSurfaceTexture = null;
        }
    }

    private void createFbo(int width, int height) {
        if (width <= 0 || height <= 0) {
            return;
        }
        if (fboTexId != 0) {
            GLES20.glDeleteTextures(1, new int[]{fboTexId}, 0);
            fboTexId = 0;
        }
        if (fboId != 0) {
            GLES20.glDeleteFramebuffers(1, new int[]{fboId}, 0);
            fboId = 0;
        }
        int[] tex = new int[1];
        GLES20.glGenTextures(1, tex, 0);
        fboTexId = tex[0];
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, fboTexId);
        GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_RGBA, width, height, 0,
                GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, null);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR);

        int[] fbo = new int[1];
        GLES20.glGenFramebuffers(1, fbo, 0);
        fboId = fbo[0];
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fboId);
        GLES20.glFramebufferTexture2D(GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0,
                GLES20.GL_TEXTURE_2D, fboTexId, 0);
        if (GLES20.glCheckFramebufferStatus(GLES20.GL_FRAMEBUFFER) != GLES20.GL_FRAMEBUFFER_COMPLETE) {
            LimeLog.warning("SGSR FBO incomplete; falling back to single-pass EASU");
            GLES20.glDeleteFramebuffers(1, new int[]{fboId}, 0);
            GLES20.glDeleteTextures(1, new int[]{fboTexId}, 0);
            fboId = 0;
            fboTexId = 0;
        }
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);
    }

    private int createExternalOESTexture() {
        int[] textures = new int[1];
        GLES20.glGenTextures(1, textures, 0);
        int id = textures[0];
        GLES20.glBindTexture(GL_TEXTURE_EXTERNAL_OES, id);
        GLES20.glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR);
        return id;
    }

    private int loadShader(int type, String src) {
        int shader = GLES20.glCreateShader(type);
        GLES20.glShaderSource(shader, src);
        GLES20.glCompileShader(shader);
        int[] compiled = new int[1];
        GLES20.glGetShaderiv(shader, GLES20.GL_COMPILE_STATUS, compiled, 0);
        if (compiled[0] == 0) {
            LimeLog.severe("SGSR shader compile failed: " + GLES20.glGetShaderInfoLog(shader));
            GLES20.glDeleteShader(shader);
            return 0;
        }
        return shader;
    }

    private int createProgram(String vertex, String fragment) {
        int vs = loadShader(GLES20.GL_VERTEX_SHADER, vertex);
        if (vs == 0) return 0;
        int fs = loadShader(GLES20.GL_FRAGMENT_SHADER, fragment);
        if (fs == 0) return 0;
        int prog = GLES20.glCreateProgram();
        if (prog == 0) return 0;
        GLES20.glAttachShader(prog, vs);
        GLES20.glAttachShader(prog, fs);
        GLES20.glLinkProgram(prog);
        int[] link = new int[1];
        GLES20.glGetProgramiv(prog, GLES20.GL_LINK_STATUS, link, 0);
        if (link[0] != GLES20.GL_TRUE) {
            LimeLog.severe("SGSR program link failed: " + GLES20.glGetProgramInfoLog(prog));
            GLES20.glDeleteProgram(prog);
            return 0;
        }
        return prog;
    }
}
