   M S B A M L   `   `   `  ���� D  @TremorApp, Version=1.0.0.0, Culture=neutral, PublicKeyToken=null%clr-namespace:TremorApp	TremorApp      TremorApp.MainWindow   \ XPresentationFramework, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35R NWindowsBase, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35W SPresentationCore, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35D 9http://schemas.microsoft.com/winfx/2006/xaml/presentation    R NSystem.Xaml, Version=4.0.0.0, Culture=neutral, PublicKeyToken=b77a5c561934e089:x,http://schemas.microsoft.com/winfx/2006/xaml     @d2http://schemas.microsoft.com/expression/blend/2008     Jmc;http://schemas.openxmlformats.org/markup-compatibility/2006     )localclr-namespace:TremorApp       � Title$  Ultimate Shaker 3000��$	��470��$	��611��.��� -    �Name$ 
resetFault�� � ColumnDefinitions �� .+��� ���	#FFF4F4F5$
��Left=�$	��396��$��
287,23,0,0q�
���   �$	��Top=�$	��142���� -   $��Connect��$
��Left=�$��
10,324,0,0q�$	��Top=�$	��272��$��20���� -    ��Name$ richTextBox1��$
��Left=�$	��181��$	��Top=�$	��120��$��149,113,0,0q�. �/� .T�J� .*�.*��� $� ���� -   $ richTextBox2��$
��Left=�$	��181��$	��Top=�$	��120��$��
11,113,0,0q�. �/� �� -   $
��Send��$
��Left=�$	��Top=�$	��271��$��
11,299,0,0q�$��20���� -   $��Activate motor��$
��Left=�$	��Top=�$	��272��$��
10,349,0,0q�$��20���� -    ��Name$ textBox1Position��$��23�� �� TextWrapping$
 Wrap=�$	��Top=�$��
299,87,0,0q�$
��Left=�$	��120���� -   $ textBox1Dec��$
��Left=�$��23��$
 Wrap=�$	��Top=�$	��120��$��297,233,0,0q��� -	   $ textBox1Acc��$��23��$
 Wrap=�$	��Top=�$��297,184,0,0q�$
��Left=�$	��120���� -
   $ textBox1Velocity��$��23��$
 Wrap=�$	��Top=�$��297,135,0,0q�$
��Left=�$	��120���� $��Position��$
��Left=�$	��Top=�$��
300,56,0,0q�$��26��$��52���� $��Velocity��$	��Top=�$��300,109,0,0q�$��26��$
��Left=�$��51�� J� RenderTransformOrigin$ 0.569,1.577&��� $��Acceleration��$	��Top=�$��300,158,0,0q�$��26��$
��Left=�$��75���� $��Deceleration��$	��Top=�$��300,207,0,0q�$��26��$
��Left=�$��77���� $��Output��$
��Left=�$	��Top=�$��
149,87,0,0q�$��26��$��48���� $��Input��$
��Left=�$	��Top=�$��	11,85,0,0q�$��26��$��38���� -   $
��Loop��$
��Left=�$��
10,374,0,0q�$	��Top=�$	��272��$��20���� -   $��	Stop loop��$
��Left=�$	��Top=�$	��272��$��
10,399,0,0q�$��20���� $��Move 1��$
��Left=�$	��Top=�$��
297,30,0,0q��� -    ��Name$ 	sendMove1��$
��Send��$
��Left=�$	��Top=�$��75��$��342,382,0,0q��� -   $ textBox1Jerk��$
��Left=�$��23��$
 Wrap=�$	��Top=�$	��120��$��300,282,0,0q��� $
��Jerk��$	��Top=�$��300,256,0,0q�$��26��$
��Left=�$��77���� -   $ textBox1JerkDec��$
��Left=�$��23��$
 Wrap=�$	��Top=�$	��120��$��300,331,0,0q��� $��	Jerkdecel��$	��Top=�$��300,305,0,0q�$��26��$
��Left=�$��77���� ���	#FFF4F4F5$
��Left=�$	��396��$��
442,23,0,0q�
���   �$	��Top=�$	��142���� -   $ textBox2Position��$��23��$
 Wrap=�$	��Top=�$��
454,87,0,0q�$
��Left=�$	��120���� -   $ textBox2Dec��$
��Left=�$��23��$
 Wrap=�$	��Top=�$	��120��$��452,233,0,0q��� -   $ textBox2Acc��$��23��$
 Wrap=�$	��Top=�$��452,184,0,0q�$
��Left=�$	��120���� -   $ textBox2Velocity��$��23��$
 Wrap=�$	��Top=�$��452,135,0,0q�$
��Left=�$	��120���� $��Position��$
��Left=�$	��Top=�$��
455,56,0,0q�$��26��$��52���� $��Velocity��$	��Top=�$��455,109,0,0q�$��26��$
��Left=�$��51��$ 0.569,1.577&��� $��Acceleration��$	��Top=�$��455,158,0,0q�$��26��$
��Left=�$��75���� $��Deceleration��$	��Top=�$��455,207,0,0q�$��26��$
��Left=�$��77���� $��Move 2��$
��Left=�$	��Top=�$��
452,30,0,0q��� -   $ textBox2Jerk��$
��Left=�$��23��$
 Wrap=�$	��Top=�$	��120��$��455,282,0,0q��� $
��Jerk��$	��Top=�$��455,256,0,0q�$��26��$
��Left=�$��77���� -   $ textBox2JerkDec��$
��Left=�$��23��$
 Wrap=�$	��Top=�$	��120��$��455,331,0,0q��� $��	Jerkdecel��$	��Top=�$��455,305,0,0q�$��26��$
��Left=�$��77���� -   $ 	sendMove2��$
��Send��$
��Left=�$	��Top=�$��75��$��499,382,0,0q��� $��11,23,330,359q�$��
Inalto.png��$
��Fill=�