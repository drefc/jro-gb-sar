#include <SPI.h>
#include <Ethernet.h>
#include <string.h>

#define DUTY_CYCLE 0.6
#define LEFT LOW
#define RIGHT HIGH

/**************************GPIO definitions**************************/

int motor_step = 9;
int motor_enable = 8;
int motor_direction = 7;
int switch_left = 5;
int switch_right = 6;
int servo=2;

/********************************************************************/

/**************************Global variables**************************/

int servo_ton;
int servo_counter;
int servo_period=2500;
int servo_width; //125-250 de -90 a 90 grados sexagesimales
int interrupt_counter=0;
long steps_moved = 0;
long steps_global = 0;
int steps_freq = 3000;
int servo_pos = 0;
String instruction = "";
//char instruction_character = '\0';
static volatile bool timer_flag = false;
static volatile bool calibration_flag = false;

/********************************************************************/

/**************************TCP configuration*************************/

byte mac[] = {0xDE, 0xAD, 0xBE, 0xEF, 0xFE, 0xED};
byte ip[] = {10, 10, 40, 245};
byte gateway[] = {10, 10, 40, 1};
byte subnet[] = {255, 255, 255, 0};
EthernetServer server = EthernetServer(12345);

/********************************************************************/

void setup()
{
  pinMode(motor_step, OUTPUT);
  pinMode(motor_direction, OUTPUT);
  pinMode(motor_enable, OUTPUT);
  //pinMode(switch_left, INPUT);
  //pinMode(switch_right, INPUT);
  pinMode(switch_left, INPUT_PULLUP);
  pinMode(switch_right, INPUT_PULLUP);
  pinMode(servo, OUTPUT);  

  digitalWrite(motor_enable, LOW);  
  Ethernet.begin(mac, ip, gateway, subnet);
  server.begin();
}

void loop()
{
  EthernetClient client = server.available();

  if (client)
    listen_instruction(client);
}

void listen_instruction(EthernetClient client)
{
  while (client.connected())
  {
    while (1)
    {
      instruction = "";
      char header = "";
      read_client(client);
      header = instruction.charAt(0);

      //move motor
      if (header == '0')
      {
        char dir = (instruction.substring(instruction.length() - 1, instruction.length() - 2)).charAt(0);
        steps_global = instruction.substring(1, instruction.length() - 2).toInt();
        calibration_flag=false;        

        if (dir == 'L')
        {          
          //move_motor(steps, LEFT, client);
          timer_flag=false;
          move_motor(LEFT, client);
          server.println("OK\n");
        }

        if (dir == 'R')
        {
          //move_motor(steps, RIGHT, client);
          timer_flag=false;
          move_motor(RIGHT, client);
          server.println("OK\n");
        }
      }

      //go to zero position on the rail
      if (header == '2')
      {
        calibration_flag = true;
        zero_position(client);
        server.println("OK\n");
        calibration_flag = false;
      }

      //end connection
      if (header == '4')  
      {
        //server.println("Bye\n");
        break;
      }

      /*      
      if (header == '5')
      {        
        servo_counter=0;
        //servo_ton=0.2;
        servo_ton=500;
        
        while(servo_counter<25)
        {
          digitalWrite(servo, HIGH);
          //delay(servo_ton);
          delayMicroseconds(servo_ton);
          digitalWrite(servo, LOW);
          //delay(20-servo_ton);
          delayMicroseconds(10000);
          delayMicroseconds(10000-servo_ton);
          servo_counter++;
        }          

        delay(400);
        servo_counter=0;
        //servo_ton=2;
        servo_ton=1800;
        
        while(servo_counter<25)
        {
          digitalWrite(servo, HIGH);
          //delay(servo_ton);
          delayMicroseconds(servo_ton);
          digitalWrite(servo, LOW);
          //delay(20-servo_ton);
          delayMicroseconds(10000);
          delayMicroseconds(10000-servo_ton);
          servo_counter++;
        }          
        
        delay(600);
        servo_counter=0;
        //servo_ton=0.2;
        servo_ton=500;
        
        while(servo_counter<25)
        { 
          digitalWrite(servo, HIGH);
          //delay(servo_ton);
          delayMicroseconds(servo_ton);
          digitalWrite(servo, LOW);
          //delay(20-servo_ton);
          delayMicroseconds(10000);
          delayMicroseconds(10000-servo_ton);
          servo_counter++;
        }
        delay(2000);        
        server.println("OK\n");
      } */      
      
      if (header == '5')
      { 
        digitalWrite(servo,HIGH);
        interrupt_counter=0;
        servo_width=250;
        timer2_on();
        delay(750);
        timer2_off();
        digitalWrite(servo,HIGH);
        interrupt_counter=0;
        servo_width=125;
        timer2_on();
        delay(3000);
        timer2_off();
        server.println("OK\n");        
      }
    }
  }
}

void move_motor(bool dir, EthernetClient client)
{
  instruction = "";
  steps_moved = 0;
  digitalWrite(motor_direction, dir);
  timer_on();

  while (1)
  {
    if (timer_flag)
    {
      timer_off();
      break;
    }
  }
}

/*
void zero_position(EthernetClient client)
{
  instruction = "";
  timer_on();
  digitalWrite(motor_direction, LEFT);

  while (1)
  {
    
      read_client(client);
      char header = instruction.charAt(0);

      if (header == '3')
      {
      timer_off();
      break;
      }
    

    if (digitalRead(switch_left) == HIGH)
    {
      timer_off();
      digitalWrite(motor_direction, RIGHT);
      timer_on();

      while (1)
      {
        if (digitalRead(switch_left) == LOW)
          break;
      }
      timer_off();
      break;
    }
  }
}
*/

void zero_position(EthernetClient client)
{
  int safety=0;
  instruction="";
  digitalWrite(motor_direction, LEFT);
  timer_on();
  
  while (1)
  {     
    if (digitalRead(switch_right)==LOW)
    {
      while (1)
      {
        if (digitalRead(switch_right)==LOW)
          safety++;
        if ((safety>=500) && (digitalRead(switch_right)==LOW))
        {
          safety=0;
          break;
        }
      }
      timer_off();      
      digitalWrite(motor_direction, RIGHT);
      timer_on();

      while(1)
      {
        if (digitalRead(switch_right)==HIGH)
            safety++;
          if ((safety>=500) && (digitalRead(switch_right)==HIGH))
          {
            safety=0;
            break;
          }
      }
      timer_off();      
      break;
    }
  } 
}

void read_client(EthernetClient client)
{
  if (client)
  {
    while (client.connected())
    {
      if (client.available())
        instruction = instruction + char(client.read());

      if (instruction.charAt(instruction.length() - 1) == '\n')
        break;
    }
  }
  return;
}

void timer_on()
{
  noInterrupts();
  TCCR1A = 0;
  TCCR1B = 0;
  TCCR1A |= (1 << COM1A1 | 1 << COM1A0 | 1 << WGM11);
  TCCR1B |= (1 << WGM12 | 1 << WGM13);
  OCR1A = (1 - DUTY_CYCLE) * steps_freq;
  ICR1 = steps_freq;
  TCCR1B |= (1 << CS10);

  if (!calibration_flag)
  {
    TIMSK1 |= (1 << OCIE1A);
    interrupts();
  }
}

void timer_off()
{
  TCCR1A = 0;
  TCCR1B = 0;
  TIMSK1 &= (0 << OCIE1A);
  noInterrupts();
}

ISR(TIMER1_COMPA_vect)
{
  if (steps_moved == steps_global)
  {
    timer_flag = true;
    timer_off();
  }
  else
  {
    steps_moved++;
  }
}

/************************************************************************/

void timer2_on()
{
  noInterrupts();
  TCCR2A=0;
  TCCR2B=0;
  TCCR2A|=(1<<WGM21);
  OCR2A=127;
  TCCR2B|=(1<<CS20);    
  TIMSK2|=(1<<OCIE2A);
  interrupts();
}

void timer2_off()
{  
  TCCR2A=0;
  TCCR2B=0;
  TIMSK2&=(0<<OCIE2A);  
  noInterrupts();
}

ISR(TIMER2_COMPA_vect)
{  
  if ((interrupt_counter==servo_width) && (digitalRead(servo)==HIGH))
  {
    digitalWrite(servo, !digitalRead(servo));    
  }

  if (interrupt_counter==(servo_period))
  {
    digitalWrite(servo, !digitalRead(servo));
    interrupt_counter=0;    
  }  
  interrupt_counter++;
}
