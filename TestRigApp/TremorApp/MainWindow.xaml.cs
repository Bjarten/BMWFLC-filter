using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.IO.Ports;
using System.Threading;
using System.Text.RegularExpressions;
//using System.Diagnostics;

namespace TremorApp
{



    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        SerialPort serialPort1 = new SerialPort(); string portName = "COM6";
        Thread thread1 = new Thread(A);
        private static EventWaitHandle ewh;
        private static EventWaitHandle ewh2;

        public MainWindow()
        {
            InitializeComponent();
            serialPort1.PortName = portName;
            serialPort1.BaudRate = 115200;
            thread1.Start(serialPort1);
            ewh = new EventWaitHandle(false, EventResetMode.ManualReset);
            ewh2 = new EventWaitHandle(false, EventResetMode.ManualReset);

            InitMoveValues();

        }

        private void InitMoveValues()
        {
            textBox1Position.Text = "0.05";
            textBox1Velocity.Text = "10";
            textBox1Acc.Text = "2000";
            textBox1Dec.Text = "2000";
            textBox1Jerk.Text = "10000";

            textBox2Position.Text = "-0.05";
            textBox2Velocity.Text = "10";
            textBox2Acc.Text = "2000";
            textBox2Dec.Text = "2000";
            textBox2Jerk.Text = "10000";
        }

        private void Button_Click(object sender, RoutedEventArgs e)
        {
            string t;
            t = serialPort1.PortName;
            sErial(t);
        }

        void sErial(String Port_name)
        {
            if (!serialPort1.IsOpen)
            {
                serialPort1.Open();
                serialPort1.WriteLine("o1.1\r");
                serialPort1.ReadTimeout = 1000;
                System.Threading.Thread.Sleep(10);
                string Connected = serialPort1.ReadExisting();
                if (Connected.Contains("Parker EME"))
                {
                    richTextBox1.Document.Blocks.Add(new Paragraph(new Run("")));
                    richTextBox1.Document.Blocks.Add(new Paragraph(new Run("Connected to " + Connected + "!")));
                  
                }

            }
        }

        private void Button_Click_2(object sender, RoutedEventArgs e)
        {
            if (serialPort1.IsOpen)
            {
                richTextBox1.Document.Blocks.Clear();
                string text = StringFromRichTextBox(richTextBox2) + "\r";
                serialPort1.WriteLine(text);
                richTextBox2.Document.Blocks.Clear();
                System.Threading.Thread.Sleep(10);
                if (serialPort1.ReadExisting().Contains(">"))
                {
                    richTextBox1.Document.Blocks.Add(new Paragraph(new Run("Manual Command Sent Successfully")));
                }
                else richTextBox1.Document.Blocks.Add(new Paragraph(new Run("Manual Command Not Accepted")));
            }
        }

        private void Button_Click_3(object sender, RoutedEventArgs e) //activate absolute positioning function and go to preset parameters.
        {
            if (serialPort1.IsOpen)
            {
                richTextBox1.Document.Blocks.Clear();
                string stop = "o1901.1=1" + "\r";
                serialPort1.WriteLine(stop);
                richTextBox1.Document.Blocks.Add(new Paragraph(new Run("")));
                System.Threading.Thread.Sleep(10);
                string msg = serialPort1.ReadExisting();
                if (serialPort1.ReadExisting().Contains(">"))
                {
                    richTextBox1.Document.Blocks.Add(new Paragraph(new Run("Activate command Sent Successfully")));
                }
                else richTextBox1.Document.Blocks.Add(new Paragraph(new Run("Activate Command Not Accepted")));
            }
        }
        string StringFromRichTextBox(RichTextBox rtb)
        {
            TextRange textRange = new TextRange(
                // TextPointer to the start of content in the RichTextBox.
                rtb.Document.ContentStart,
                // TextPointer to the end of content in the RichTextBox.
                rtb.Document.ContentEnd
            );

            // The Text property on a TextRange object returns a string
            // representing the plain text content of the TextRange.
            return textRange.Text;
        }

        private void Button_Click_4(object sender, RoutedEventArgs e)
        {
            if (serialPort1.IsOpen)
            {
                ewh.Set();
            }
        }

        private static void A(object parameter)
        {
            SerialPort serialPort1 = parameter as SerialPort;
            string adr1 = "o1100.3=24835" + "\r";
            string adr2 = "o1100.3=25091" + "\r";
            string reset = "o1100.3=16387" + "\r";
            string status = "o1000.3" + "\r";


            string numberString = string.Empty;
            string msg = string.Empty;

            Boolean positionReached = false;
            Boolean isMoving = false;

            while (true)
            {
                ewh2.Set();
                ewh.WaitOne();
                ewh2.Reset();


                serialPort1.WriteLine(reset);
                System.Threading.Thread.Sleep(10);
                serialPort1.WriteLine(adr1);
                System.Threading.Thread.Sleep(10);

                while (!positionReached)
                {   
                    serialPort1.WriteLine(status);
                    System.Threading.Thread.Sleep(10);
                    msg = serialPort1.ReadExisting();
                    //Debug.WriteLine(msg);
                    numberString = Regex.Match(msg, @"\d+").Value;

                    if (numberString == "256")
                        isMoving = true;

                    if ((numberString == "2304") && isMoving)
                    {
                        isMoving = false;
                        positionReached = true;
                    }
                }
                positionReached = false;

                serialPort1.WriteLine(reset);
                System.Threading.Thread.Sleep(10);
                serialPort1.WriteLine(adr2);


                while (!positionReached)
                {
                    serialPort1.WriteLine(status);
                    System.Threading.Thread.Sleep(10);
                    msg = serialPort1.ReadExisting();
                    //Debug.WriteLine(msg);
                    numberString = Regex.Match(msg, @"\d+").Value;

                    if (numberString == "256")
                        isMoving = true;

                    if ((numberString == "2304") && isMoving)
                    {
                        isMoving = false;
                        positionReached = true;
                    }
                }
                positionReached = false;

                System.Threading.Thread.Sleep(10);
            }
        }

        private void Button_Click_5(object sender, RoutedEventArgs e)
        {   
            ewh.Reset();
            ewh2.WaitOne();
            string stop = "o1100.3=0" + "\r";
            string start = "o1100.3=1" + "\r";
            string stopAdr = "o1100.3=25347" + "\r";
            string reset = "o1100.3=16387" + "\r";

            System.Threading.Thread.Sleep(10);
            serialPort1.WriteLine(reset);
            System.Threading.Thread.Sleep(10);
            serialPort1.WriteLine(start);
            System.Threading.Thread.Sleep(10);
            serialPort1.WriteLine(stop);

            if (serialPort1.ReadExisting().Contains(">"))
            {
                richTextBox1.Document.Blocks.Add(new Paragraph(new Run("Stop command Sent Successfully")));
            }
            else richTextBox1.Document.Blocks.Add(new Paragraph(new Run("Stop Command Not Accepted")));
        }

        private void sendMove1_Click(object sender, RoutedEventArgs e)
        {
            string position = "o1901.1=" + textBox1Position.Text + "\r";
            string velocity = "o1902.1=" + textBox1Velocity.Text + "\r";
            string acceleration = "o1906.1=" +  textBox1Acc.Text + "\r";
            string deceleration = "o1907.1=" + textBox1Dec.Text + "\r";
            string jerk = "o1908.1=" + textBox1Jerk.Text + "\r";

            if (serialPort1.IsOpen)
            {
                richTextBox1.Document.Blocks.Clear();

                serialPort1.WriteLine(position);
                System.Threading.Thread.Sleep(50);
                if (serialPort1.ReadExisting().Contains(">"))
                {
                    richTextBox1.Document.Blocks.Add(new Paragraph(new Run("Move 1 position Command Sent Successfully")));
                }
                else richTextBox1.Document.Blocks.Add(new Paragraph(new Run("position Command Not Accepted")));

                serialPort1.WriteLine(velocity);
                System.Threading.Thread.Sleep(50);
                if (serialPort1.ReadExisting().Contains(">"))
                {
                    richTextBox1.Document.Blocks.Add(new Paragraph(new Run("velocity Command Sent Successfully")));
                }
                else richTextBox1.Document.Blocks.Add(new Paragraph(new Run("velocity Command Not Accepted")));

                serialPort1.WriteLine(acceleration);
                System.Threading.Thread.Sleep(50);
                if (serialPort1.ReadExisting().Contains(">"))
                {
                    richTextBox1.Document.Blocks.Add(new Paragraph(new Run("acceleratio Command Sent Successfully")));
                }
                else richTextBox1.Document.Blocks.Add(new Paragraph(new Run("acceleratio Command Not Accepted")));

                serialPort1.WriteLine(deceleration);
                System.Threading.Thread.Sleep(50);
                if (serialPort1.ReadExisting().Contains(">"))
                {
                    richTextBox1.Document.Blocks.Add(new Paragraph(new Run("decelerationCommand Sent Successfully")));
                }
                else richTextBox1.Document.Blocks.Add(new Paragraph(new Run("deceleration Command Not Accepted")));

                serialPort1.WriteLine(jerk);
                System.Threading.Thread.Sleep(50);
                if (serialPort1.ReadExisting().Contains(">"))
                {
                    richTextBox1.Document.Blocks.Add(new Paragraph(new Run("jerk Command Sent Successfully")));
                }
                else richTextBox1.Document.Blocks.Add(new Paragraph(new Run("jerk Command Not Accepted")));
            }
        }

        private void sendMove2_Click(object sender, RoutedEventArgs e)
        {
            string position = "o1901.2=" + textBox2Position.Text + "\r";
            string velocity = "o1902.2=" + textBox2Velocity.Text + "\r";
            string acceleration = "o1906.2=" + textBox2Acc.Text + "\r";
            string deceleration = "o1907.2=" + textBox2Dec.Text + "\r";
            string jerk = "o1908.2=" + textBox2Jerk.Text + "\r";

            if (serialPort1.IsOpen)
            {
                richTextBox1.Document.Blocks.Clear();

                serialPort1.WriteLine(position);
                System.Threading.Thread.Sleep(50);
                if (serialPort1.ReadExisting().Contains(">"))
                {
                    richTextBox1.Document.Blocks.Add(new Paragraph(new Run("Move 2 position Command Sent Successfully")));
                }
                else richTextBox1.Document.Blocks.Add(new Paragraph(new Run("position Command Not Accepted")));

                serialPort1.WriteLine(velocity);
                System.Threading.Thread.Sleep(50);
                if (serialPort1.ReadExisting().Contains(">"))
                {
                    richTextBox1.Document.Blocks.Add(new Paragraph(new Run("velocity Command Sent Successfully")));
                }
                else richTextBox1.Document.Blocks.Add(new Paragraph(new Run("velocity Command Not Accepted")));

                serialPort1.WriteLine(acceleration);
                System.Threading.Thread.Sleep(50);
                if (serialPort1.ReadExisting().Contains(">"))
                {
                    richTextBox1.Document.Blocks.Add(new Paragraph(new Run("acceleratio Command Sent Successfully")));
                }
                else richTextBox1.Document.Blocks.Add(new Paragraph(new Run("acceleratio Command Not Accepted")));

                serialPort1.WriteLine(deceleration);
                System.Threading.Thread.Sleep(50);
                if (serialPort1.ReadExisting().Contains(">"))
                {
                    richTextBox1.Document.Blocks.Add(new Paragraph(new Run("decelerationCommand Sent Successfully")));
                }
                else richTextBox1.Document.Blocks.Add(new Paragraph(new Run("deceleration Command Not Accepted")));

                serialPort1.WriteLine(jerk);
                System.Threading.Thread.Sleep(50);
                if (serialPort1.ReadExisting().Contains(">"))
                {
                    richTextBox1.Document.Blocks.Add(new Paragraph(new Run("jerk Command Sent Successfully")));
                }
                else richTextBox1.Document.Blocks.Add(new Paragraph(new Run("jerk Command Not Accepted")));

            }
        }


    }


}




