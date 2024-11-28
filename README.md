**阅读目录(Content)**

* [1\.AmplifyImpostorInspector部分](https://github.com)
+ [1\.1 RenderCombinedAlpha](https://github.com)
+ [1\.2 GenerateAutomaticMesh](https://github.com)

* [2\.AmplifyImpostor部分](https://github.com)
+ [2\.1 Remapping](https://github.com)

* [3\.Shader渲染部分](https://github.com)
+ [3\.1 SphereImpostorVertex](https://github.com)
+ [3\.2 SphereImpostorFragment](https://github.com):[PodHub豆荚加速器官方网站](https://rikeduke.com)

首先看一下点击Bake按钮后的执行流程：


![](https://img2024.cnblogs.com/blog/519009/202411/519009-20241127134034288-420618995.jpg)


# 1\.AmplifyImpostorInspector部分


首先点击按钮设置了bakeTexture \= true




```
if( GUILayout.Button( TextureIcon, "buttonright", GUILayout.Height( 24 ) ) )
{
    // now recalculates texture and mesh every time because mesh might have changed
    //if( m_instance.m_alphaTex == null )
    //{
        m_outdatedTexture = true;
        m_recalculatePreviewTexture = true;
    //}

    bakeTextures = true;
}
```


 


如果展开了BillboardMesh选项或是bakeTextures为true，则都会执行下面部分：




```
if( ( ( m_billboardMesh || m_recalculatePreviewTexture ) && m_instance.m_alphaTex == null ) || ( bakeTextures && m_recalculatePreviewTexture ) )
{
    try
    {
        m_instance.RenderCombinedAlpha( m_currentData );
    }
    catch( Exception e )
    {
        Debug.LogWarning( "[AmplifyImpostors] Something went wrong with the mesh preview process, please contact support@amplify.pt with this log message.\n" + e.Message + e.StackTrace );
    }

    if( m_instance.m_cutMode == CutMode.Automatic )
        m_recalculateMesh = true;
    m_recalculatePreviewTexture = false;
}
```


 


## 1\.1 RenderCombinedAlpha


该函数会遍历一遍所有视角的模型，生成出覆盖范围最大的Bounds，并更新到这2个变量中：




```
m_xyFitSize = Mathf.Max(m_xyFitSize, frameBounds.size.x, frameBounds.size.y);
m_depthFitSize = Mathf.Max(m_depthFitSize, frameBounds.size.z);
```


通过RenderImpostor函数的combinedAlphas变量，将所有视角模型的alpha叠加在一张RT上，再通过这张叠加RT


修正原有Bounds：




```
m_xyFitSize *= maxBound;
m_depthFitSize *= maxBound;
```


 


接着得到哪张材质的索引对应传入RT集合的alpha材质：




```
bool standardRendering = m_data.Preset.BakeShader == null;
int alphaIndex = m_data.Preset.AlphaIndex;
if (standardRendering && m_renderPipelineInUse == RenderPipelineInUse.HDRP)
    alphaIndex = 3;
else if (standardRendering)
    alphaIndex = 2;
```


 


用深度图的边缘生成alpha：




```
RenderTexture tempTex = RenderTextureEx.GetTemporary(m_alphaGBuffers[3]);
Graphics.Blit(m_alphaGBuffers[3], tempTex);
packerMat.SetTexture("_A", tempTex);
Graphics.Blit(m_trueDepth, m_alphaGBuffers[3], packerMat, 11);
RenderTexture.ReleaseTemporary(tempTex);
```


shader:




```
Pass // copy depth 11
{
    ZTest Always Cull Off ZWrite Off

    CGPROGRAM
    #pragma target 3.0
    #pragma vertex vert_img
    #pragma fragment frag
    #include "UnityCG.cginc"

    uniform sampler2D _MainTex;
    uniform sampler2D _A;

    float4 frag( v2f_img i ) : SV_Target
    {
        float depth = SAMPLE_RAW_DEPTH_TEXTURE( _MainTex, i.uv ).r;
        float3 color = tex2D( _A, i.uv ).rgb;
        float alpha = 1 - step( depth, 0 );

        return float4( color, alpha );
    }
    ENDCG
}
```


合并后的alpha会单独存下来，也就是每一个sheet格子的alpha叠在一起，这样做可以让最终生成面片的顶点合理覆盖：


![](https://img2024.cnblogs.com/blog/519009/202411/519009-20241127151251133-2046305675.jpg)


 


## 1\.2 GenerateAutomaticMesh


这个函数主要生成顶点，会存到AmplifyImpostorAsset的ShapePoints中。


顶点数据会给接下来的GenerateMesh使用。


 


这一步一定会设上triangulateMesh \= true;




```
if (m_recalculateMesh && m_instance.m_alphaTex != null)
{
    m_recalculateMesh = false;
    m_instance.GenerateAutomaticMesh(m_currentData);
    triangulateMesh = true;
    EditorUtility.SetDirty(m_instance);
}
```


 


接着设置previewMesh：




```
if (triangulateMesh)
    m_previewMesh = GeneratePreviewMesh(m_currentData.ShapePoints, true);
```


 


然后会将CutMode改为手动，允许用户二次修改：




```
if (autoChangeToManual /*&& Event.current.type == EventType.Layout*/ )
{
    autoChangeToManual = false;
    m_instance.m_cutMode = CutMode.Manual;
    Event.current.Use();
}
```


最后进入DelayedBake，调用AmplifyImpostor的RenderAllDeferredGroups函数。


 


# 2\.AmplifyImpostor部分


进入函数RenderAllDeferredGroups，前面都和之前操作差不多，直到调用到RenderImpostor：




```
if (impostorMaps)
{
    commandBuffer.SetViewProjectionMatrices(V, P);
    commandBuffer.SetViewport(new Rect((m_data.TexSize.x / hframes) * x, (m_data.TexSize.y / (vframes + (impostorType == ImpostorType.Spherical ? 1 : 0))) * y, (m_data.TexSize.x / m_data.HorizontalFrames), (m_data.TexSize.y / m_data.VerticalFrames)));
```


绘制时每个sheet的格子都存放对应角度的模型图片，通过SetViewport进行绘制目标区域的裁剪。


不同的ImpostorType对应绘制hframes、vframes的排布方式也不一样。


 


绘制代码基本的逻辑结构如下：




```
for (int x = 0; x < hframes; x++)
{
    for (int y = 0; y <= vframes; y++)
    {
        if (impostorMaps)
        {
            commandBuffer.SetViewProjectionMatrices(V, P);
            commandBuffer.SetViewport(new Rect((m_data.TexSize.x / hframes) * x, (m_data.TexSize.y / (vframes + (impostorType == ImpostorType.Spherical ? 1 : 0))) * y, (m_data.TexSize.x / m_data.HorizontalFrames), (m_data.TexSize.y / m_data.VerticalFrames)));

            if (standardrendering && m_renderPipelineInUse == RenderPipelineInUse.HDRP)
            {
                commandBuffer.SetGlobalMatrix("_ViewMatrix", V);
                commandBuffer.SetGlobalMatrix("_InvViewMatrix", V.inverse);
                commandBuffer.SetGlobalMatrix("_ProjMatrix", P);
                commandBuffer.SetGlobalMatrix("_ViewProjMatrix", P * V);
                commandBuffer.SetGlobalVector("_WorldSpaceCameraPos", Vector4.zero);
            }
        }

        for (int j = 0; j < validMeshesCount; j++)
        {
            commandBuffer.DrawRenderer...
        }
    }
}
Graphics.ExecuteCommandBuffer(commandAlphaBuffer);
```


优先绘制Y轴，其次X轴，每次绘制写入commandBuffer，最后统一在外部执行ExecuteCommandBuffer。


附一张测试例图方便参考：


![](https://img2024.cnblogs.com/blog/519009/202411/519009-20241127171153713-507670403.png)


 


## 2\.1 Remapping


这一步工作主要是将深度通道塞进去。


 


合并Alpha：




```
// Switch alpha with occlusion
RenderTexture tempTex = RenderTexture.GetTemporary(m_rtGBuffers[0].width, m_rtGBuffers[0].height, m_rtGBuffers[0].depth, m_rtGBuffers[0].format);
RenderTexture tempTex2 = RenderTexture.GetTemporary(m_rtGBuffers[3].width, m_rtGBuffers[3].height, m_rtGBuffers[3].depth, m_rtGBuffers[3].format);

packerMat.SetTexture("_A", m_rtGBuffers[2]);
Graphics.Blit(m_rtGBuffers[0], tempTex, packerMat, 4); //A.b
packerMat.SetTexture("_A", m_rtGBuffers[0]);
Graphics.Blit(m_rtGBuffers[3], tempTex2, packerMat, 4); //B.a
Graphics.Blit(tempTex, m_rtGBuffers[0]);
Graphics.Blit(tempTex2, m_rtGBuffers[3]);
RenderTexture.ReleaseTemporary(tempTex);
RenderTexture.ReleaseTemporary(tempTex2);
```


 


shader:




```
Pass // Copy Alpha 4
{
    CGPROGRAM
    #pragma target 3.0
    #pragma vertex vert_img
    #pragma fragment frag
    #include "UnityCG.cginc"

    uniform sampler2D _MainTex;
    uniform sampler2D _A;

    fixed4 frag (v2f_img i ) : SV_Target
    {
        float alpha = tex2D( _A, i.uv ).a;
        fixed4 finalColor = (float4(tex2D( _MainTex, i.uv ).rgb , alpha));
        return finalColor;
    }
    ENDCG
}
```


 


这一步会将RT\[2]的alpha合并至RT\[0]，将RT\[0]的alpha合并至RT\[3]


 


接下来PackDepth，将深度信息写入RT\[2]的A通道：




```
// Pack Depth
PackingRemapping(ref m_rtGBuffers[2], ref m_rtGBuffers[2], 0, packerMat, m_trueDepth);
m_trueDepth.Release();
m_trueDepth = null;
```


 


RT\[2]存的是法线，a通道存深度后：


![](https://img2024.cnblogs.com/blog/519009/202411/519009-20241127154257195-1378897876.jpg)


 


RT\[0]的alpha：


![](https://img2024.cnblogs.com/blog/519009/202411/519009-20241127154450924-319425116.jpg)


 


 


FixAlbedo，m\_rtGBuffers\[1]对应extraTex参数，若传参会被设置到\_A采样器。




```
// Fix Albedo
PackingRemapping(ref m_rtGBuffers[0], ref m_rtGBuffers[0], 5, packerMat, m_rtGBuffers[1]);
```


alb.rgb / (1\-spec)不太清楚。




```
Pass // Fix albedo 5
{
    CGPROGRAM
    #pragma target 3.0
    #pragma vertex vert_img
    #pragma fragment frag
    #include "UnityCG.cginc"

    uniform sampler2D _MainTex;
    uniform sampler2D _A; //specular

    fixed4 frag (v2f_img i ) : SV_Target
    {
        float3 spec = tex2D( _A, i.uv ).rgb;
        float4 alb = tex2D( _MainTex, i.uv );
        alb.rgb = alb.rgb / (1-spec);
        return alb;
    }
    ENDCG
}
```


 


存TGA（如果预设里勾选了TGA则调用该处，否则存PNG）：




```
// TGA
for (int i = 0; i < outputList.Count; i++)
{
    if (outputList[i].ImageFormat == ImageFormat.TGA)
        PackingRemapping(ref m_rtGBuffers[i], ref m_rtGBuffers[i], 6, packerMat);
}
```


 


DilateShader边缘膨胀处理：




```
Shader dilateShader = AssetDatabase.LoadAssetAtPath(AssetDatabase.GUIDToAssetPath(DilateGUID));
Debug.Log(dilateShader, dilateShader);
Material dilateMat = new Material(dilateShader);

// Dilation
for (int i = 0; i < outputList.Count; i++)
{
    if (outputList[i].Active)
        DilateRenderTextureUsingMask(ref m_rtGBuffers[i], ref m_rtGBuffers[alphaIndex], m_data.PixelPadding, alphaIndex != i, dilateMat);
}
```


 


shader是沿着周围8个方向外拓一圈：




```
float4 frag_dilate( v2f_img i, bool alpha )
{
    float2 offsets[ 8 ] =
    {
        float2( -1, -1 ),
        float2(  0, -1 ),
        float2( +1, -1 ),
        float2( -1,  0 ),
        float2( +1,  0 ),
        float2( -1, +1 ),
        float2(  0, +1 ),
        float2( +1, +1 )
    };
```


 


函数中会根据pixelBlend将这个shader调用N次：




```
for (int i = 0; i < pixelBleed; i++)
{
    dilateMat.SetTexture("_MaskTex", dilatedMask);

    Graphics.Blit(mainTex, tempTex, dilateMat, alpha ? 1 : 0);
    Graphics.Blit(tempTex, mainTex);

    Graphics.Blit(dilatedMask, tempMask, dilateMat, 1);
    Graphics.Blit(tempMask, dilatedMask);
}
```


 


默认值是调用32次：




```
[SerializeField]
[Range( 0, 64 )]
public int PixelPadding = 32;
```


 


 


# 3\.Shader渲染部分


![](https://img2024.cnblogs.com/blog/519009/202411/519009-20241127161704803-303452137.jpg)


Octahedron八面体方案和球面分别使用2个对外Shader，


八面体方案会采样3次做插值，球面则代码稍少，接下来只看球面部分。


 


## 3\.1 SphereImpostorVertex


先看ForwardBase的pass：


顶点部分执行SphereImpostorVertex( v.vertex, v.normal, o.frameUVs, o.viewPos );


这个函数会处理Billboard的位置信息，并返回常规顶点信息和frameUVs信息。


得到相对相机位置，并转换至object空间，\_Offset是实际模型中心偏移量，通过像素转顶点的方式离线计算得到




```
float3 objectCameraPosition = mul( ai_WorldToObject, float4( worldCameraPos, 1 ) ).xyz - _Offset.xyz; //ray origin
float3 objectCameraDirection = normalize( objectCameraPosition );
```


构建一组基向量：




```
float3 upVector = float3( 0,1,0 );
float3 objectHorizontalVector = normalize( cross( objectCameraDirection, upVector ) );
float3 objectVerticalVector = cross( objectHorizontalVector, objectCameraDirection );
```


横向信息用arctan2，变量名作者写错了




```
float verticalAngle = frac( atan2( -objectCameraDirection.z, -objectCameraDirection.x ) * AI_INV_TWO_PI ) * sizeX + 0.5;
```


纵向信息用acos将点乘转线性




```
float verticalDot = dot( objectCameraDirection, upVector );
float upAngle = ( acos( -verticalDot ) * AI_INV_PI ) + axisSizeFraction * 0.5f;
```


yRot构建的旋转矩阵用作细节修正




```
float yRot = sizeFraction.x * AI_PI * verticalDot * ( 2 * frac( verticalAngle ) - 1 );

// Billboard rotation
float2 uvExpansion = vertex.xy;
float cosY = cos( yRot );
float sinY = sin( yRot );
float2 uvRotator = mul( uvExpansion, float2x2( cosY, -sinY, sinY, cosY ) );
```


最后sizeFraction用于将坐标缩放为对应sheet内格子大小




```
float2 frameUV = ( ( uvExpansion * fractionsUVscale + 0.5 ) + relativeCoords ) * sizeFraction;
```


## 3\.2 SphereImpostorFragment


frag一些逻辑都是常规操作，看下深度部分的处理，


离近了看会有真实深度的遮挡：


![](https://img2024.cnblogs.com/blog/519009/202411/519009-20241128095200202-800499755.jpg)


 


因为是正交相机拍摄，不存在DeviceDepth转线性EyeDepth。


 


深度赋值取的clipPos.z：




```
fixed4 frag_surf (v2f_surf IN, out float outDepth : SV_Depth ) : SV_Target {
    ...
    IN.pos.zw = clipPos.zw;
    outDepth = IN.pos.z;
```


 


\_DepthSize读的是csharp变量m\_depthFitSize，在烘焙时这个值是正交相机的远截面：




```
Matrix4x4 P = Matrix4x4.Ortho(-fitSize + m_pixelOffset.x, fitSize + m_pixelOffset.x, -fitSize + m_pixelOffset.y, fitSize + m_pixelOffset.y, 0, zFar: -m_depthFitSize);
```


最后深度计算这里，\_DepthSize\*0\.5猜测是物体中心是z\=0\.5，是基于物体中心增加偏移深度，并且remapNormal.a之前已经随着法线做了\-1 \- 1的映射操作：




```
float4 remapNormal = normalSample * 2 - 1; // object normal is remapNormal.rgb
```


最后乘以length( ai\_ObjectToWorld\[ 2 ].xyz )其实是乘以Z轴的缩放，如果没有缩放改成1结果不变：




```
float depth = remapNormal.a * _DepthSize * 0.5 * length( ai_ObjectToWorld[ 2 ].xyz );
```


 


计算完后再将颜色和深度输出：




```
fixed4 frag_surf (v2f_surf IN, out float outDepth : SV_Depth ) : SV_Target {
    UNITY_SETUP_INSTANCE_ID(IN);
    SurfaceOutputStandardSpecular o;
    UNITY_INITIALIZE_OUTPUT( SurfaceOutputStandardSpecular, o );

    float4 clipPos;
    float3 worldPos;
    SphereImpostorFragment( o, clipPos, worldPos, IN.frameUVs, IN.viewPos );
    IN.pos.zw = clipPos.zw;

    outDepth = IN.pos.z;

    UNITY_APPLY_DITHER_CROSSFADE(IN.pos.xy);
    return float4( _ObjectId, _PassValue, 1.0, 1.0 );
}
```


 


阴影部分ShadowCaster pass用了同样的代码，因此impostor也有阴影。


