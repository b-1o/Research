#!/usr/bin/env python
#coding:utf-8
from httphandler import Request
import pickle
import math

######## html #########
html_body ="""
<html>
    <head>
        <meta charset="UTF-8">
        <title>CNN Application%s</title>
        <style>
            #body{width:100%%;height:100%%}
            #leftsideDiv{width:70%%;height:100%%;float:left}
            #fileDiv{width:100%%;height:25%%}
            #svgDiv{width:100%%;height:75%%}
            #rightsideDiv{width:30%%;height:100%%;float:left}

            #classTd{width:10%%;text-align:left}
            #classNumTd{width:15%%;text-align:left}
            #classMaxTd{width:50%%;text-align:left}

            #layerNumTd{width:5%%;text-align:center}
            #layerSelectTd{width:20%%}
            #layerParamTd{width:75%%}
        </style>
    </head>

    <body id="body">
        <div id="leftsideDiv">
            <div id="fileDiv">
                <div style="width:50%%;height:100%%;float:left">
                    <h1>Parameter</h1>
                </div>
                <div style="width:50%%;height:100%%;float:left;overflow:auto">
                    <h1>Classes</h1>
                    <table>
                        <tr>
                            <td id="classTd">Class0
                            </td>
                            <td id="classNumTd">
                                <input name="class0" />
                            </td>
                            <td id="classMaxTd">
                                <div>test</div>
                            </td>
                        </tr>
                        <tr>
                            <td id="classTd">Class1
                            </td>
                            <td id="classNumTd">
                                <input name="class1" />
                            </td>
                        </tr>
                        <tr>
                            <td id="classTd">Class2
                            </td>
                            <td id="classNumTd">
                                <input name="class2" />
                            </td>
                        </tr>
                        <tr>
                            <td id="classTd">Class3
                            </td>
                            <td id="classNumTd">
                                <input name="class3" />
                            </td>
                        </tr>
                        <tr>
                            <td id="classTd">Class4
                            </td>
                            <td id="classNumTd">
                                <input name="class4" />
                            </td>
                        </tr>
                        <tr>
                            <td id="classTd">Class5
                            </td>
                            <td id="classNumTd">
                                <input name="class5" />
                            </td>
                        </tr>
                        <tr>
                            <td id="classTd">Class6
                            </td>
                            <td id="classNumTd">
                                <input name="class6" />
                            </td>
                        </tr>
                        <tr>
                            <td id="classTd">Class7
                            </td>
                            <td id="classNumTd">
                                <input name="class7" />
                            </td>
                        </tr>
                        <tr>
                            <td id="classTd">Class8
                            </td>
                            <td id="classNumTd">
                                <input name="class8" />
                            </td>
                        </tr>
                        <tr>
                            <td id="classTd">Class9
                            </td>
                            <td id="classNumTd">
                                <input name="class9" />
                            </td>
                        </tr>
                    </table>
                </div>
            </div>
            <div id="svgDiv">
                <div style="width:100%%;height:10%%">
                    <h1>Diagram</h1>
                </div>
                <div style="width:100%%;height:90%%;overflow:auto">

                    <!-- diagram div -->
                    <div style="width:%s%%;height:100%%">
                        %s
                    </div>

                </div>
            </div>
        </div>
        <div id="rightsideDiv" style="overflow:auto">
            <h1>Layers</h1>
            <form method="POST" action="/cgi-bin/gui.py" name="networkForm" id="networkForm">
                <table>
                    <tr>
                        <td>
                            <input type="button" id=addrow value="Add"onClick="addLayer();">
                        </td>
                        <td>
                            <input type="button" id=delLayer value="Remove" onClick="deleteLayer();" disabled="true">
                        </td>
                        <td>
                            <input type="submit" value="submit" onClick="return_layer_num();">
                        </td>
                    </tr>
                </table>

                <table style="width:100%%">
                    <tbody id="networkTable">
                        <tr>
                            <td id="layerNumTd">
                            </td>
                            <td id="layerSelectTd">
                                <select id="selectData" name="data" onchange="changeClassNum();" required>
                                    <option></option>
                                    <option value="Mnist">Mnist</option>
                                    <option value="Cifar">Cifar</option>
                                    <option value="Uniqlo">Uniqlo</option>
                                </select>
                            </td>
                            <td id="layerParamTd">
                            </td>
                        </tr>
                        <tr>
                            <td id="layerNumTd">1
                            </td>
                            <td id="layerSelectTd">
                                <select id="select1" name="layer1" onchange="changeDiv(1);" required>
                                    <option></option>
                                    <option value="Convolution">Convolution</option>
                                    <option value="Pooling">Pooling    </option>
                                    <option value="Relu">Relu       </option>
                                    <option value="Affine">Affine   </option>
                                    <option value="Softmax">Softmax </option>
                                    <option value="BatchNorm">BatchNorm</option>
                                </select>
                            </td>
                            <td id="layerParamTd">
                                <div id="Conv1" style="display:none">
                                    <input id="Channel1" name="Channel1" placeholder="Channel" style="width:30%%"/>
                                    <input id="Width1" name="Width1"    placeholder="Width"  style="width:30%%"/>
                                    <input id="Height1" name="Height1"   placeholder="Height" style="width:30%%"/>
                                </div>
                                <div id="Affine1" style="display:none">
                                    <input name="Output1" style="width:30%%" placeholder="Output"/>
                                </div>
                            </td>
                        </tr>
                    </tbody>
                </table>

                <input type="hidden" name="layer_num" id ="layer_num" value="0">
            </form>
        </div>
    </body>

    <script language="JavaScript">

    function changeDiv(i){
        selecter = 'select' + i;
        if(document.getElementById(selecter)){
            id = document.getElementById(selecter).value;
            ConvDiv = 'Conv' + i;
            AffineDiv = 'Affine' + i;
            if(id == "Convolution"){
                document.getElementById(ConvDiv).style.display = "";
                document.getElementById(AffineDiv).style.display = "none";
            }else if(id == "Affine"){
                document.getElementById(ConvDiv).style.display = "none";
                document.getElementById(AffineDiv).style.display = "";
            }else{
                document.getElementById(ConvDiv).style.display = "none";
                document.getElementById(AffineDiv).style.display = "none";
            }
        }
    }
    function initDiv(loop){
        for(i = 1; i < loop; i++){
            changeDiv(i);
        }
    }

    layer_num = 1
    function addLayer(){
        layer_num++;
        var networkTable = document.getElementById("networkTable");

            var current_row  = document.createElement("TR" );
            current_row.setAttribute("id", "layer"+layer_num);

                // レイヤ数を表示
                current_td = document.createElement("TD");
                current_td.setAttribute("id", "layerNumTd");
                currenttext=document.createTextNode(layer_num);
                current_td.appendChild(currenttext);
                current_row.appendChild(current_td);

                // レイヤをセレクト
                current_td = document.createElement("TD");
                current_td.setAttribute("id", "layerSelectTd");
                current_td.setAttribute("onchange", "changeDiv(" +layer_num+ ");");
                //onchange="changeDiv(1);"
                //current_row.appendChild(current_td);
                    // セレクトボタン
                    current_form=document.createElement("SELECT");
                    current_form.setAttribute("id","select" + layer_num);
                    current_form.setAttribute("name", "layer" + layer_num);
                    current_form.setAttribute("required", "");
                    var selectBox = {
                      "":"",
                      "Convolution" :"Convolution",
                      "Pooling"     :"Pooling",
                      "Relu"        :"Relu",
                      "Affine"      :"Affine",
                      "Softmax"     :"Softmax",
                      "BatchNorm"   :"BatchNorm"
                    };
                    for(var j in selectBox){
                      var option = document.createElement("option");
                      option.setAttribute("value", j);
                      option.innerHTML = selectBox[j];
                      current_form.appendChild(option)
                    }
                    current_td.appendChild(current_form);

                current_row.appendChild(current_td);

                //パラメータを入力
                current_td = document.createElement("TD");
                current_td.setAttribute("id", "layerParamTd");

                    current_div = document.createElement("DIV");
                    current_div.setAttribute("id", "Conv" + layer_num);
                    current_div.setAttribute("style","display:none")

                        // チャネル入力
                        current_form=document.createElement("INPUT");
                        current_form.setAttribute("placeholder", "Channel");
                        current_form.setAttribute("type", "NUMBER");
                        //current_form.setAttribute("id", "inputParam");
                        current_form.setAttribute("style", "width:30%%");
                        current_form.setAttribute("id", "Channel" + layer_num);
                        current_form.setAttribute("name", "Channel" + layer_num)
                        current_div.appendChild(current_form);

                        // 横幅入力
                        current_form=document.createElement("INPUT");
                        current_form.setAttribute("placeholder", "Width");
                        current_form.setAttribute("type", "NUMBER");
                        //current_form.setAttribute("id", "inputParam");
                        current_form.setAttribute("style", "width:30%%");
                        current_form.setAttribute("id", "Width" + layer_num);
                        current_form.setAttribute("name", "Width" + layer_num)
                        current_div.appendChild(current_form);

                        // 縦幅入力
                        current_form=document.createElement("INPUT");
                        current_form.setAttribute("placeholder", "Height");
                        current_form.setAttribute("type", "NUMBER");
                        //current_form.setAttribute("id", "inputParam");
                        current_form.setAttribute("style", "width:30%%");
                        current_form.setAttribute("id", "Height" + layer_num);
                        current_form.setAttribute("name", "Height" + layer_num)
                        current_div.appendChild(current_form);

                    current_td.appendChild(current_div);

                    current_div = document.createElement("DIV");
                    current_div.setAttribute("id", "Affine" + layer_num);
                    current_div.setAttribute("style","display:none")

                        // 縦幅入力
                        current_form=document.createElement("INPUT");
                        current_form.setAttribute("placeholder", "Output");
                        current_form.setAttribute("type", "NUMBER");
                        //current_form.setAttribute("id", "inputParam");
                        current_form.setAttribute("style", "width:30%%");
                        current_form.setAttribute("id", "Output" + layer_num);
                        current_form.setAttribute("name", "Output" + layer_num)
                        current_div.appendChild(current_form);

                    current_td.appendChild(current_div);

                current_row.appendChild(current_td);

            networkTable.appendChild(current_row);

        document.networkForm.delLayer.disabled=false;
    }

    function initLayer(){
        for(var i = 0; i < %s-1; i++){
            addLayer(i);
        }
    }
    window.onload = initLayer();

    function deleteLayer(){
        var networkTable = document.getElementById("networkTable");
        var removeLayer  = document.getElementById("layer"+layer_num);
        networkTable.removeChild(removeLayer);
        layer_num--;

        if(layer_num == 1){
            document.networkForm.delLayer.disabled=true;
        }
    }

    function return_layer_num(){
        document.getElementById("layer_num").value = layer_num;

        //document.getElementById("selectData").value1.selected = True;
        //window.document.form.select.option2.selected = true;
    }

    function select_data(){
        document.getElementById("selectData").value = "%s";
        if(!document.getElementById("select2")){
            document.getElementById("select1").value = "Default";
        }else{
            %s
        }

    }
    window.onload = select_data();
    window.onload = initDiv(%s);



    </script>
</html>
"""

select_layer = """
    if(document.getElementById("select%s")){
        document.getElementById("select%s").value = "%s";
        %s
    }
"""

select_conv_layer = """
    document.getElementById("Channel%s").value = %s;
    document.getElementById("Width%s").value = %s;
    document.getElementById("Height%s").value = %s;
"""
select_affine_layer = """
    document.getElementById("Output%s").value = %s;
"""

one_layer = """
<div style="width:%s%%;height:100%%;float:left">
    <svg id="svg" style="width:100%%;height:70%%;background-color:white">
        %s
    </svg>
    <div style="width:100%%;height:30%%;background-color:white">
        <h3 style="margin:0px;text-align:center">%s</h3>
            %s
    </div>
</div>
"""
img = """
    <img src="Cifar10.png">
"""
svg_rect = """
    <rect x="%s%%" y="%s%%" width="%s%%" height="%s%%" stroke-width="0.5%%" stroke="black" fill="white"/>
"""
svg_circle = """
    <circle cx="50%%" cy="%s%%" r="7%%" stroke="black" fill="white" stroke-width="0.1%%"/>
"""
one_layer_3param = """
    <div style="text-align:center">Channel : %s</div>
    <div style="text-align:center">Width   : %s</div>
    <div style="text-align:center">Heigth  : %s</div>
"""
one_layer_1param = """
    <div style="text-align:center">%s : %s</div>
"""

one_arrow = """
<div style="width:%s%%;height:100%%;float:left">
    <svg id="arrow" style="width:100%%;height:70%%;background-color:white">
        <defs>
            <marker id="mu_mh" markerUnits="strokeWidth" markerWidth="3" markerHeight="3" viewBox="0 0 10 10" refX="5" refY="5">
                <polygon points="0,0 5,5 0,10 10,5 " fill="black"/>
            </marker>
        </defs>
        <g stroke="black" >
            <line x1="20%%" y1="50%%" x2="80%%" y2="50%%" stroke-width="4" marker-end="url(#mu_mh)"/>
        </g>
    </svg>

    <div style="width:100%%;height:30%%;background-color:white">
        <h3 style="margin:0px;text-align:center">%s</h3>
            %s
    </div>
</div>
"""
conv_arrow = """
    <div style="text-align:center">InputChannel  : %s</div>
    <div style="text-align:center">FilterChannel : %s</div>
    <div style="text-align:center">Width         : %s</div>
    <div style="text-align:center">Heigth        : %s</div>
"""



# ネットワーク構成
network_list = ["Input"]
# パラメータを格納するディクショナリ
params_dict  =     { "InputData" : "Default" }
params_dict.update({ "LayerNum"  : 0 })


# リクエストを取得
req = Request()

# データの種類を取得
if req.form.has_key("data"):
    params_dict["InputData"] = req.form["data"].value

    if params_dict["InputData"] == "Mnist":
        params_dict.update({"InputChannel":1})
        params_dict.update({"InputWidth":28})
        params_dict.update({"InputHeight":28})

    elif params_dict["InputData"] == "Cifar":
        params_dict.update({"InputChannel":3})
        params_dict.update({"InputWidth":32})
        params_dict.update({"InputHeight":32})

    elif params_dict["InputData"] == "Uniqlo":
        params_dict.update({"InputChannel":3})
        params_dict.update({"InputWidth":50})
        params_dict.update({"InputHeight":50})
        #params_dict.update({"InputChannel":3})
        #params_dict.update({"InputWidth":28})
        #params_dict.update({"InputHeight":28})

# レイヤ数を取得
if req.form.has_key("layer_num"):
    params_dict["LayerNum"] = int(req.form["layer_num"].value)

# チャネル等の初期値
if params_dict["LayerNum"] != 0:
    pre_channel = params_dict["InputChannel"]
    pre_width = params_dict["InputWidth"]
    pre_height = params_dict["InputHeight"]
    pre_neuron = 0
    rate = 50.0 / float( pre_width )

# ネットワーク構成を取得
for i in range( params_dict["LayerNum"] ):
    network_list.append( req.form["layer" + str(i+1) ].value )

params_dict.update({"NetworkList" : network_list})

# ネットワークのパラメータを取得
for i in range( params_dict["LayerNum"] ):
    if network_list[i+1] == "Convolution":
        params_dict.update({"Channel" + str(i+1) : int(req.form["Channel" + str(i+1)].value)})
        params_dict.update({"Width"   + str(i+1) : int(req.form["Width"   + str(i+1)].value)})
        params_dict.update({"Height"  + str(i+1) : int(req.form["Height"  + str(i+1)].value)})
    if network_list[i+1] == "Affine":
        params_dict.update({"Output"  + str(i+1) : int(req.form["Output"  + str(i+1)].value)})



# ダイアグラムの横幅、reshape考慮
diagram_width = 100+60*( params_dict["LayerNum"] )
layer_width = 100 * (2.0/float(2+3*(params_dict["LayerNum"]+1)))
arrow_width = 100 * (1.0/float(2+3*(params_dict["LayerNum"]+1)))

# 入力レイヤ
if params_dict["LayerNum"] != 0:
    parameter = ( params_dict["InputChannel"], params_dict["InputWidth"], params_dict["InputHeight"] )
    input_param = one_layer_3param%parameter

    # 出力レイヤの図を生成
    square = "%s" * pre_channel
    rect = [0] * pre_channel
    rect_all = ()
    width_size   = rate * pre_width
    height_size  = rate * pre_height
    width_start  = ( 100 - width_size )  / 2.0 - ( pre_channel / 2.0 )
    height_start = ( 100 - height_size ) / 2.0 - ( pre_channel / 2.0 )
    for j in range(pre_channel):
        rect[j] = svg_rect % ( width_start + j, height_start + j, width_size, height_size )
        rect_all += ( rect[j], )
    square = square % rect_all

    parameter = (layer_width, square, params_dict["InputData"], input_param )
    input_layer = one_layer%parameter


# ダイアグラム作成
diagram = "%s"
content = ""
all_layer = [0] * params_dict["LayerNum"]
all_arrow = [0] * params_dict["LayerNum"]
reshape = ()

# 一回目のAffine層だけフラグを立てる(Reshape層を入れるため)、もっとうまい方法があるきがする
affine_flag = False
affine_cnt = 0

for i in range( params_dict["LayerNum"] ):
    if network_list[i+1] == "Convolution":
        # パラメータ取得
        channel = params_dict["Channel"+str(i+1)]
        width   = params_dict["Width"+str(i+1)]
        height  = params_dict["Height"+str(i+1)]

        conv_param = conv_arrow % ( pre_channel, channel, width, height )

        pre_channel  = channel
        pre_width    = pre_width - width + 1
        pre_height   = pre_height - height + 1

        parameter    = (arrow_width, network_list[i+1], conv_param)
        all_arrow[i] = one_arrow%parameter

        parameter = ( pre_channel, pre_width, pre_height )
        layer_param = one_layer_3param%parameter

        # 出力レイヤの図を生成
        square = "%s" * pre_channel
        rect = [0] * pre_channel
        rect_all = ()
        width_size   = rate * pre_width
        height_size  = rate * pre_height
        width_start  = ( 100 - width_size )  / 2.0 - ( pre_channel / 2.0 )
        height_start = ( 100 - height_size ) / 2.0 - ( pre_channel / 2.0 )
        for j in range(pre_channel):
            rect[j] = svg_rect % ( width_start + j, height_start + j, width_size, height_size )
            rect_all += ( rect[j], )
        square = square % rect_all

        parameter = (layer_width, square, "Layer" + str(i+1),layer_param)
        all_layer[i] = one_layer%parameter

    elif network_list[i+1] == "Affine":
        if affine_cnt == 0 : pre_neuron = pre_channel * pre_width * pre_height

        parameter = ( "Neuron", pre_neuron )
        layer_param = one_layer_1param % parameter

        if pre_neuron <= 5000:
            # 出力レイヤの図を生成
            circle = "%s" * pre_neuron
            en = [0] * pre_neuron
            en_all = ()
            for j in range(pre_neuron):
                en[j] = svg_circle % (10+(80/float(pre_neuron-1))*j)
                en_all += ( en[j], )
            circle = circle % en_all
        else:
            circle = '<text x="20%" y="50%" fill="black">Too many neurons</text>'

        parameter = ( layer_width, circle, "Reshape Layer", layer_param )
        reshape_layer = one_layer%parameter
        parameter = (arrow_width, "Reshape", "")
        reshape_arrow = one_arrow%parameter
        reshape = ( reshape_arrow, reshape_layer )
        affine_flag = True

        neuron = params_dict["Output"+str(i+1)]
        layer_param = one_layer_1param % ( "Neuron", neuron )

        if neuron <= 5000:
            # 出力レイヤの図を生成
            circle = "%s" * neuron
            en = [0] * neuron
            en_all = ()
            for j in range(neuron):
                en[j] = svg_circle % (10+(80/float(neuron-1))*j)
                en_all += ( en[j], )
            circle = circle % en_all
        else:
            circle = '<text x="20%" y="50%" fill="black">Too many neurons</text>'

        parameter = ( layer_width, circle, "Layer"+str(i+1), layer_param )
        all_layer[i] = one_layer%parameter
        parameter = (arrow_width, network_list[i+1], "")
        all_arrow[i] = one_arrow%parameter

    elif network_list[i+1] == "Pooling":
        pre_width = int( math.ceil( pre_width / 2 ) );
        pre_height /= 2;
        parameter = ( pre_channel, pre_width, pre_height )
        layer_param = one_layer_3param % parameter

        # 出力レイヤの図を生成
        square = "%s" * pre_channel
        rect = [0] * pre_channel
        rect_all = ()
        width_size   = rate * pre_width
        height_size  = rate * pre_height
        width_start  = ( 100 - width_size )  / 2.0 - ( pre_channel / 2.0 )
        height_start = ( 100 - height_size ) / 2.0 - ( pre_channel / 2.0 )
        for j in range(pre_channel):
            rect[j] = svg_rect % ( width_start + j, height_start + j, width_size, height_size )
            rect_all += ( rect[j], )
        square = square % rect_all

        parameter = (layer_width, square, "Layer" + str(i+1), layer_param)
        all_layer[i] = one_layer%parameter
        parameter = (arrow_width, network_list[i+1], "")
        all_arrow[i] = one_arrow%parameter

    elif network_list[i+1] == "Relu":
        if affine_cnt == 0:
            parameter = ( pre_channel, pre_width, pre_height )
            layer_param = one_layer_3param % parameter

            # 出力レイヤの図を生成
            square = "%s" * pre_channel
            rect = [0] * pre_channel
            rect_all = ()
            width_size   = rate * pre_width
            height_size  = rate * pre_height
            width_start  = ( 100 - width_size )  / 2.0 - ( pre_channel / 2.0 )
            height_start = ( 100 - height_size ) / 2.0 - ( pre_channel / 2.0 )
            for j in range(pre_channel):
                rect[j] = svg_rect % ( width_start + j, height_start + j, width_size, height_size )
                rect_all += ( rect[j], )
            square = square % rect_all

            parameter = (layer_width, square, "Layer" + str(i+1), layer_param)
            all_layer[i] = one_layer%parameter
            parameter = (arrow_width, network_list[i+1], "")
            all_arrow[i] = one_arrow%parameter
        else:
            if( params_dict.get("Output"+str(i), 0 ) != 0 ):
                output = params_dict.get("Output"+str(i))
            else:
                output = params_dict["Output"+str(i-1)]
            layer_param = one_layer_1param % ( "Neuron",neuron )

            if neuron <= 5000:
                # 出力レイヤの図を生成
                circle = "%s" * neuron
                en = [0] * neuron
                en_all = ()
                for j in range(neuron):
                    en[j] = svg_circle % (10+(80/float(neuron-1))*j)
                    en_all += ( en[j], )
                circle = circle % en_all
            else:
                circle = '<text x="20%" y="50%" fill="black">Too many neurons</text>'

            parameter = (layer_width, circle, "Layer" + str(i+1), layer_param)
            all_layer[i] = one_layer%parameter
            parameter = (arrow_width, network_list[i+1], "")
            all_arrow[i] = one_arrow%parameter

    elif network_list[i+1] == "Softmax":
        if( params_dict.get("Output"+str(i), 0 ) != 0 ):
            output = params_dict.get("Output"+str(i))
        else:
            output = params_dict["Output"+str(i-1)]
        layer_param = one_layer_1param % ( "Class", output )

        if output <= 5000:
            # 出力レイヤの図を生成
            circle = "%s" * output
            en = [0] * output
            en_all = ()
            for j in range(output):
                en[j] = svg_circle % (10+(80/float(output-1))*j)
                en_all += ( en[j], )
            circle = circle % en_all
        else:
            circle = '<text x="20%" y="50%" fill="black">Too many classes</text>'

        parameter = (layer_width, circle, "Layer" + str(i+1), layer_param )
        all_layer[i] = one_layer%parameter
        parameter = (arrow_width, network_list[i+1], "")
        all_arrow[i] = one_arrow%parameter

    else:
        parameter = (layer_width, "", "Layer" + str(i+1), "")
        all_layer[i] = one_layer%parameter
        parameter = (arrow_width, network_list[i+1], "")
        all_arrow[i] = one_arrow%parameter


    if i == 0:
        if affine_flag and affine_cnt == 0:
            parameter = ( input_layer, ) + reshape + ( all_arrow[i], all_layer[i] )
            diagram = "%s%s%s%s%s"
            content = parameter
            affine_flag = False
            affine_cnt += 1

        else:
            parameter = ( input_layer, all_arrow[i], all_layer[i] )
            diagram = "%s%s%s"
            content = parameter
    else:
        if affine_flag and affine_cnt == 0:
            parameter = reshape + ( all_arrow[i], all_layer[i] )
            diagram += "%s%s%s%s"
            content += parameter
            affine_flag = False
            affine_cnt += 1

        else:
            parameter = ( all_arrow[i], all_layer[i] )
            diagram += "%s%s"
            content += parameter

diagram = diagram%content

parameter = ()
layer = "%s" * params_dict["LayerNum"]
for i in range(params_dict["LayerNum"]):

    if network_list[i+1] == "Convolution":
        content = select_conv_layer % ( i+1, params_dict["Channel"+str(i+1)], i+1, params_dict["Width"+str(i+1)], i+1, params_dict["Height"+str(i+1)] )
    elif network_list[i+1] == "Affine":
        content = select_affine_layer % ( i+1, params_dict["Output"+str(i+1)] )
    else:
        content = ""

    current_layer = select_layer % ( i+1, i+1, network_list[i+1], content )
    parameter += ( current_layer, )


layer = layer % parameter

# BatchNormalization の対応
for i in range(params_dict["LayerNum"]):
    if network_list[i] == "BatchNorm":
        if network_list[i-1] != "Convolution":
            params_dict["Output"+str(i)] = params_dict["Output"+str(i-1)]

# パラメータ保存
with open("parameter.pkl", 'wb') as file:
    pickle.dump(params_dict, file, -1)

print "Content-type: text/html\n"
parameter =( 1, diagram_width, diagram, params_dict["LayerNum"], params_dict["InputData"], layer, params_dict["LayerNum"])
print html_body%parameter

# パラメータ保存
with open("parameter.pkl", 'wb') as file:
    pickle.dump(params_dict, file, -1)
