/*
 * Copyright (C) 2011-2014 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * This file is auto-generated. DO NOT MODIFY!
 * The source Renderscript file: D:\\20240926study\\20240926junior\\hello_ar_java\\hello_ar_java\\app\\src\\main\\rs\\yuv420888.rs
 */

package com.google.ar.core.examples.java.Tree;

import android.renderscript.*;
import com.google.ar.core.examples.java.Tree.yuv420888BitCode;

/**
 * @hide
 */
public class ScriptC_yuv420888 extends ScriptC {
    private static final String __rs_resource_name = "yuv420888";
    // Constructor
    public  ScriptC_yuv420888(RenderScript rs) {
        super(rs,
              __rs_resource_name,
              yuv420888BitCode.getBitCode32(),
              yuv420888BitCode.getBitCode64());
        __I32 = Element.I32(rs);
        __U32 = Element.U32(rs);
        __ALLOCATION = Element.ALLOCATION(rs);
        __U8_4 = Element.U8_4(rs);
    }

    private Element __ALLOCATION;
    private Element __I32;
    private Element __U32;
    private Element __U8_4;
    private FieldPacker __rs_fp_ALLOCATION;
    private FieldPacker __rs_fp_I32;
    private FieldPacker __rs_fp_U32;
    private final static int mExportVarIdx_width = 0;
    private int mExportVar_width;
    public synchronized void set_width(int v) {
        setVar(mExportVarIdx_width, v);
        mExportVar_width = v;
    }

    public int get_width() {
        return mExportVar_width;
    }

    public Script.FieldID getFieldID_width() {
        return createFieldID(mExportVarIdx_width, null);
    }

    private final static int mExportVarIdx_height = 1;
    private int mExportVar_height;
    public synchronized void set_height(int v) {
        setVar(mExportVarIdx_height, v);
        mExportVar_height = v;
    }

    public int get_height() {
        return mExportVar_height;
    }

    public Script.FieldID getFieldID_height() {
        return createFieldID(mExportVarIdx_height, null);
    }

    private final static int mExportVarIdx_picWidth = 2;
    private long mExportVar_picWidth;
    public synchronized void set_picWidth(long v) {
        if (__rs_fp_U32!= null) {
            __rs_fp_U32.reset();
        } else {
            __rs_fp_U32 = new FieldPacker(4);
        }
        __rs_fp_U32.addU32(v);
        setVar(mExportVarIdx_picWidth, __rs_fp_U32);
        mExportVar_picWidth = v;
    }

    public long get_picWidth() {
        return mExportVar_picWidth;
    }

    public Script.FieldID getFieldID_picWidth() {
        return createFieldID(mExportVarIdx_picWidth, null);
    }

    private final static int mExportVarIdx_uvPixelStride = 3;
    private long mExportVar_uvPixelStride;
    public synchronized void set_uvPixelStride(long v) {
        if (__rs_fp_U32!= null) {
            __rs_fp_U32.reset();
        } else {
            __rs_fp_U32 = new FieldPacker(4);
        }
        __rs_fp_U32.addU32(v);
        setVar(mExportVarIdx_uvPixelStride, __rs_fp_U32);
        mExportVar_uvPixelStride = v;
    }

    public long get_uvPixelStride() {
        return mExportVar_uvPixelStride;
    }

    public Script.FieldID getFieldID_uvPixelStride() {
        return createFieldID(mExportVarIdx_uvPixelStride, null);
    }

    private final static int mExportVarIdx_uvRowStride = 4;
    private long mExportVar_uvRowStride;
    public synchronized void set_uvRowStride(long v) {
        if (__rs_fp_U32!= null) {
            __rs_fp_U32.reset();
        } else {
            __rs_fp_U32 = new FieldPacker(4);
        }
        __rs_fp_U32.addU32(v);
        setVar(mExportVarIdx_uvRowStride, __rs_fp_U32);
        mExportVar_uvRowStride = v;
    }

    public long get_uvRowStride() {
        return mExportVar_uvRowStride;
    }

    public Script.FieldID getFieldID_uvRowStride() {
        return createFieldID(mExportVarIdx_uvRowStride, null);
    }

    private final static int mExportVarIdx_ypsIn = 5;
    private Allocation mExportVar_ypsIn;
    public synchronized void set_ypsIn(Allocation v) {
        setVar(mExportVarIdx_ypsIn, v);
        mExportVar_ypsIn = v;
    }

    public Allocation get_ypsIn() {
        return mExportVar_ypsIn;
    }

    public Script.FieldID getFieldID_ypsIn() {
        return createFieldID(mExportVarIdx_ypsIn, null);
    }

    private final static int mExportVarIdx_uIn = 6;
    private Allocation mExportVar_uIn;
    public synchronized void set_uIn(Allocation v) {
        setVar(mExportVarIdx_uIn, v);
        mExportVar_uIn = v;
    }

    public Allocation get_uIn() {
        return mExportVar_uIn;
    }

    public Script.FieldID getFieldID_uIn() {
        return createFieldID(mExportVarIdx_uIn, null);
    }

    private final static int mExportVarIdx_vIn = 7;
    private Allocation mExportVar_vIn;
    public synchronized void set_vIn(Allocation v) {
        setVar(mExportVarIdx_vIn, v);
        mExportVar_vIn = v;
    }

    public Allocation get_vIn() {
        return mExportVar_vIn;
    }

    public Script.FieldID getFieldID_vIn() {
        return createFieldID(mExportVarIdx_vIn, null);
    }

    //private final static int mExportForEachIdx_root = 0;
    private final static int mExportForEachIdx_doConvert = 1;
    public Script.KernelID getKernelID_doConvert() {
        return createKernelID(mExportForEachIdx_doConvert, 58, null, null);
    }

    public void forEach_doConvert(Allocation aout) {
        forEach_doConvert(aout, null);
    }

    public void forEach_doConvert(Allocation aout, Script.LaunchOptions sc) {
        // check aout
        if (!aout.getType().getElement().isCompatible(__U8_4)) {
            throw new RSRuntimeException("Type mismatch with U8_4!");
        }
        forEach(mExportForEachIdx_doConvert, (Allocation) null, aout, null, sc);
    }

}

