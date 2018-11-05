var username=prompt('Enter username')
// $.ajax({
//   type: "GET",
//   url: "http://127.0.0.1:5000/username/"+username,
//   response = HttpResponse(json_data, content_type='application/json'),
//   success: function(data){
//     console.log('User details'+data)
//   }

//   });

// var socket = io.connect('http://127.0.0.1:5000');
// socket.emit('username', username)
var out_l;
var txt=" [More]";
var nxt=" [Next doc match]";
var welcome_text="Hello, how can I help you,";
var download="[download]";
var filename='something.txt';
var votingBool=true;
//var count=0;

//var count =0;

var $messages = $('.messages-content'),
  d, h, m,
  i=0;
  $(window).load(function() {

    $messages.mCustomScrollbar();
    welcome(username,welcome_text);
});


  // socket.on('imageConversionByServer', function(data) {

  //   var figure=new Image();
  //   figure.src=data

  //   $('<div class="message new"><figure class="avatar"><img src="dribbble-01_1x.jpg" /></figure>'+ '</div>').append(figure).appendTo($('.mCSB_container')).addClass('new');
  //   console.log(figure)
  // });


function updateScrollbar() {
  $messages.mCustomScrollbar("update").mCustomScrollbar('scrollTo', 'bottom', {
    scrollInertia: 10,
    timeout: 0
  });
}

function setDate(){
  d = new Date()
  if (m != d.getMinutes()) {
    m = d.getMinutes();
    $('<div class="timestamp">' + d.getHours() + ':' + m + '</div>').appendTo($('.message:last'));
  }
}

function insertMessage() {
  msg = $('.message-input').val();
  window.count=0;
  window.query_count=0;
  var flag=0
  $.ajax({
    type: "POST",


    url: "http://127.0.0.1:5000/input/"+msg+flag,
    data: JSON.stringify("{'username':'" + username+"','query':'"+msg+"'}"),
    dataType: 'json',
    success: function(data){
      $('.message.loading').remove();
      fakeMessage(data)
    },
    error: function (data){
          $('.message.loading').remove();
          fakeMessage(data);
    }
    },
    console.log(msg,flag)
 );
  // socket.emit('input', msg,flag,username)
  if ($.trim(msg) == '') {
    return false;
  }
  $('<div class="message message-personal">' + msg + '</div>').appendTo($('.mCSB_container')).addClass('new');
  setDate();
  $('.message-input').val(null);
  $('<div class="message loading new"><figure class="avatar"><img src="dribbble-01_1x.jpg" /></figure><span></span></div>').appendTo($('.mCSB_container'));
  updateScrollbar();
  setTimeout(function() {
  }, 1000 + (Math.random() * 20) * 100);
}

$('.message-submit').click(function() {
  insertMessage();
});

$(window).on('keydown', function(e) {
  if (e.which == 13) {
    insertMessage();
    return false;
  }
});
//function voting(rel_para,rel_loc, output1){
//console.log(output1[1],"+++++++")
//if(votingBool){
//$('<div class="votestamp"><div docId = output1  rel_loc = rel_loc  class="sprite sprite-fa-thumbs-up-grey" > '+'</div>').appendTo($('.message:last'));
////  $('<div class="votestamp"><div class="sprite sprite-fa-thumbs-up-grey" > '+'</div>').appendTo($('.message.chat-response-cont'));
//}
//
//
//
//
//
//$(".sprite-fa-thumbs-up-grey").click(function(){
//
//    rel_loc = $(this).attr( " rel_loc " );
//    docId = $(this).attr( " output1[1] " );
//    rel_para[rel_loc]+=1;
//    $.ajax({
//      type: "POST",
//      data: JSON.stringify("{'rel_para':'" + rel_para +"', 'output1[1]': '"+docId+"', 'username':'" + username+"', 'name':'" + msg+"'}"),
//      dataType: 'json',
//      url: "http://127.0.0.1:5000/relevance",
//
//      }),
//      console.log('data sent')
//    }
//    );
//    }

//

function voting(rel_para,rel_loc, doc_id,msg){
console.log(rel_para,"rel para")
console.log(rel_loc)
if(votingBool){



  $('<div class="votestamp"><div docId ="'+ doc_id+'" rel_loc = "'+ rel_loc+'"  msg="'+msg+'" id="'+ doc_id+ rel_loc+'" class="sprite sprite-fa-thumbs-up-grey" > '+'</div>').appendTo($('.message.chat-response-cont'));
//    rel_loc = $(this).attr( "rel_loc" );
//    console.log(rel_loc,"hdfgvjd")


$(".sprite-fa-thumbs-up-grey").click(function(){
      var re1_para=[0,0,0]

    rel_loc = $(this).attr( "rel_loc" );
    msg = $(this).attr( "msg" );
    id = $(this).attr( "id" );
    console.log(rel_loc)
    console.log(id,"iddd")
    docId = $(this).attr( "docId" );
    console.log(docId,"doc")
    rel_para[rel_loc]+=1;
    console.log(rel_para,"reee")
    $.ajax({
      type: "POST",
      data: JSON.stringify("{'rel_para':'" + rel_para +"', 'docId': '"+doc_id+"', 'username':'" + username+"', 'name':'" + msg+"'}"),
      dataType: 'json',
      url: "http://127.0.0.1:5000/relevance",

      }),
      console.log('data sent')
    }
    );
    }
    }


//
//function voting(rel_para,rel_loc){
//console.log(rel_para,"rel para")
//console.log(rel_loc,"rel loc")
//  console.log("click happened");
//if(votingBool){
//  $('<div class="votestamp"><div class="sprite sprite-fa-thumbs-up-grey" > '+'</div>').appendTo($('.message.chat-response-cont'));
//}
//
//
// $(".sprite-fa-thumbs-up-grey").click(function(){
//
//    console.log("like button")
////
//    rel_para[rel_loc]+=1;
//    $.ajax({
//      type: "POST",
//      data: JSON.stringify("{'rel_para':'" + rel_para +"', 'username':'" + username+"', 'name':'" + msg+"'}"),
//      dataType: 'json',
//      url: "http://127.0.0.1:5000/relevance",
//      }),
//      console.log('data sent')
//    }
//    // socket.emit('relevance',rel_para);
//    );
//    }

function welcome(username,welcome_text){
  console.log('in welcome',username)

  $('<div class="message new"><figure class="avatar"><img src="dribbble-01_1x.jpg" /></figure>'+ "Hello "+ username +", how can I help you?" +'</div>').appendTo($('.mCSB_container')).addClass('new');
  updateScrollbar();
}


function fakeMessage(output1,doc_id) {
  updateScrollbar();
  if ($('.message-input').val() != '') {
    return false;
  }
  $('<div class="message loading new"><figure class="avatar"><img src="dribbble-01_1x.jpg" /></figure><span></span></div>').appendTo($('.mCSB_container'));
  console.log(doc_id,"docid");
//  window.output1[1]
  console.log(output1,"oooo");
  console.log(output1[1],"outtt")
  console.log(output1[0][0],"outputt")
  console.log(output1[0],"len")
  if (output1[0]== "Oops ! Sorry, I am unable to answer to this query. ")
  {
    votingBool=false;
    Fake=JSON.stringify(output1[0].replace(/(\r\n|\n|\r)/gm," "));
    console.log(Fake)
     $('<div class="message new"><figure class="avatar"><img src="dribbble-01_1x.jpg" /></figure>'+ Fake +'</div>').appendTo($('.mCSB_container')).addClass('new');
        updateScrollbar();
        $('.message.loading').remove();
    return false;
    }

  else{
  out_l=output1[0].length
//  console.log(out_1,"output111")
  console.log("output length: ",out_l)
  // flag=0
  Fake=JSON.stringify(output1[0][0].replace(/(\r\n|\n|\r)/gm," "));
  // console.log(out_l)
  console.log(Fake)
  }



  if (out_l>2) {
    Fake2=JSON.stringify(output1[0][1].replace(/(\r\n|\n|\r)/gm," "));
    Fake3=JSON.stringify(output1[0][2].replace(/(\r\n|\n|\r)/gm," "));



    }
    setTimeout(() => {

      var alpha=[0,0,0]
      var theta=0;
      console.log('in timeout')
      console.log('fake in timeout:',Fake)
      $('.message.loading').remove();

      if(out_l>2){

        $('<div class="message new more-first chat-response-cont"><figure class="avatar"><img src="dribbble-01_1x.jpg" /></figure>'+ Fake + '<span class="more-first ">'+ txt + '</span></div>').appendTo($('.mCSB_container')).addClass('new');
        $('<div class="message new more-second chat-response-cont dispNone"><figure class="avatar"><img src="dribbble-01_1x.jpg" /></figure>'+ Fake2 + '<span class="more2">'+ txt + '</span></div>').appendTo($('.mCSB_container')).addClass('new');
        $('<div class="message new next-div chat-response-cont dispNone"><figure class="avatar"><img src="dribbble-01_1x.jpg" /></figure>'+ Fake3 + '<span class="next-span">'+ nxt + '</span><span class="download-span" docId="'+ output1[1]+'">'+ download + '</span> </div>').appendTo($('.mCSB_container')).addClass('new');

         theta=0;
        votingBool=true;
        voting(alpha,theta,output1[1],msg);
        console.log('Voting completed + more loaded')
        updateScrollbar();

        $("span.more-first").on('click',function(){
          theta=1;
          votingBool=true;
          voting(alpha,theta,output1[1],msg);
          updateScrollbar();
          $("div.more-second").removeClass("dispNone")
          console.log('Voting completed+more1');
            });

          $("div.more-second span.more2").on('click',function(){
            theta=2;
            votingBool=true;
            voting(alpha,theta,output1[1],msg);
            updateScrollbar();
        $("div.next-div").removeClass("dispNone")
        console.log('Voting completed+more2');
        });

         $("div.next-div span.download-span").on('click',function(){
//         count=window.count;
//         query_count=0;
//         query_count++;
         console.log(msg+query_count,"msg")
         window.location="http://127.0.0.1:5000/downloads?doc_id="+$(this).attr( "docid" );
         });
//




         $("div.next-div span.next-span").on('click',function(){
           count=window.count;
           count++;
             $('<div class="message loading new"><figure class="avatar"><img src="dribbble-01_1x.jpg" /></figure><span></span></div>').appendTo($('.mCSB_container'));
              console.log('button click');
              console.log(msg,'message');
//               count++;
               console.log(count,"flag");
              $.ajax({
                type: "POST",
                url: "http://127.0.0.1:5000/input/"+msg+count,
                data: JSON.stringify("{'username':'" + username+"','query':'"+msg+"'}"),
                dataType: 'json',
                success: function(data){

                  $('.message.loading').remove();
                  fakeMessage(data)
                },
                error: function(data){
                }

              }),
              updateScrollbar();

                console.log(msg,count);

              // socket.emit('input', msg,flag,username )
              // console.log('socket emit log, flag:',flag)
              // fakeMessage();

      });
      }



      else if (out_l==1){

        $('<div class="message new"><figure class="avatar"><img src="dribbble-01_1x.jpg" /></figure>'+ Fake + '<next>' + nxt + '</next>' + '</div>').appendTo($('.mCSB_container')).addClass('new');
        console.log('in else')
        var theta=0;
        var alpha=[0,0,0]
        voting(alpha,theta,output1[1],msg);
        updateScrollbar();
        $("next").on('click',function(click){
          $('<div class="message loading new"><figure class="avatar"><img src="dribbble-01_1x.jpg" /></figure><span></span></div>').appendTo($('.mCSB_container'));
          var flag=1;
          $.ajax({
            type: "POST",
            url: "http://127.0.0.1:5000/input/"+msg+flag,
            data: JSON.stringify("{'username':'" + username+"','query':'"+msg+"'}"),
            dataType: 'json',
            success: function(data){
              $('.message.loading').remove();
              fakeMessage(data)
            }
          })
          });


      }
      else {


        $('<div class="message new"><figure class="avatar"><img src="dribbble-01_1x.jpg" /></figure>'+ Fake +'</div>').appendTo($('.mCSB_container')).addClass('new');
        updateScrollbar();
      }
      $('.message.loading').remove();
    },1000)
     updateScrollbar();

}

