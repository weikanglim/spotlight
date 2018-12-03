import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';
import { HttpClientModule } from '@angular/common/http';
import { FormsModule }   from '@angular/forms';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { ClassifierComponent } from './classifier/classifier.component';
import { HomeComponent } from './home/home.component';
import { DataComponent } from './data/data.component';
import { TrainComponent } from './train/train.component';
import { FileDropModule } from 'ngx-file-drop';
import { TestComponent } from './test/test.component';
import { AgGridModule } from 'ag-grid-angular';
import { NgbModule } from '@ng-bootstrap/ng-bootstrap';


@NgModule({
  declarations: [
    AppComponent,
    ClassifierComponent,
    HomeComponent,
    DataComponent,
    TrainComponent,
    TestComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    FileDropModule,
    HttpClientModule,
    FormsModule,
    AgGridModule.withComponents([]),
    NgbModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
